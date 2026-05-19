"""
UGCP network based on a 2D SwinUNETR backbone.

This implementation follows the 3D UGCP computation and only changes the
spatial operators from 3D to 2D.
"""

from monai.networks.nets import SwinUNETR
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinUNETR2D_UGCP(nn.Module):
    """
    2D SwinUNETR with UGCP refinement for binary segmentation.
    """

    def __init__(
        self,
        in_channels=1,
        backbone_feat_out_ch=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        act="RELU",
        cfg=None,
        **kwargs,
    ):
        super().__init__()

        if cfg is not None:
            mcfg = cfg.model
            ugcp_steps = mcfg.ugcp_steps
            ugcp_eta = mcfg.ugcp_eta
            ugcp_u0 = mcfg.ugcp_u0
            ugcp_tau = mcfg.ugcp_tau
            self.use_ugcp = mcfg.use_ugcp
            self.ugcp_use_source_term = mcfg.ugcp_use_source_term
        else:
            ugcp_steps = 2
            ugcp_eta = 0.3
            ugcp_u0 = 0.5
            ugcp_tau = 0.1
            self.use_ugcp = True
            self.ugcp_use_source_term = True

        self.backbone = SwinUNETR(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=backbone_feat_out_ch,
            feature_size=24,
            patch_size=2,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4.0,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            dropout_path_rate=0.1,
            norm_name="instance",
            use_checkpoint=False,
        )

        self.logit_head = nn.Conv2d(
            backbone_feat_out_ch,
            2,
            kernel_size=1,
            bias=True,
        )
        self.feat_head = nn.Conv2d(
            backbone_feat_out_ch,
            2,
            kernel_size=1,
            bias=True,
        )

        if self.use_ugcp:
            self.ugcp = UQFluxRefine2D(
                K_steps=ugcp_steps,
                eta=ugcp_eta,
                u0=ugcp_u0,
                tau=ugcp_tau,
                source_item=self.ugcp_use_source_term,
                feat_head_channel=2,
            )

        print("SwinUNETR2D_UGCP initialized. UGCP:", self.use_ugcp)

    def forward(self, x):
        backbone_features = self.backbone(x)
        aligned_feat = self.feat_head(backbone_features)
        logits = self.logit_head(backbone_features)

        if self.use_ugcp:
            refined_logits = self.ugcp(logits, aligned_feat)
        else:
            refined_logits = logits

        return refined_logits


class UQFluxRefine2D(nn.Module):
    """
    Uncertainty-gated conservative refinement on binary 2D logits.
    """

    def __init__(
        self,
        K_steps=2,
        eta=0.3,
        u0=0.5,
        tau=0.1,
        source_item=True,
        feat_head_channel=2,
    ):
        super().__init__()
        self.K_steps = K_steps
        self.eta = eta
        self.u0 = u0
        self.tau = tau
        self.ugcp_use_source_term = source_item

        self.stencil = DepthwiseStencil2D(1)
        self.stencil_f = DepthwiseStencil2D(feat_head_channel)
        self.edge_mlp = EdgeMLP2D(feat_head_channel)

    def forward(self, logits, aligned_feat):
        refined_logits = logits

        for _ in range(self.K_steps):
            # alpha = softplus(s) + 1, pi = alpha / sum(alpha), u = K / sum(alpha).
            _, center_uncertainty, _ = evidential_prob_unc(refined_logits)
            foreground_logit = refined_logits[:, 1:2]

            neighbor_foreground_logits = self.stencil(foreground_logit)
            batch_size, num_neighbors, channels, height, width = (
                neighbor_foreground_logits.shape
            )
            neighbor_fg_flat = neighbor_foreground_logits.view(
                batch_size * num_neighbors, channels, height, width
            )

            neighbor_logits = torch.cat(
                [torch.zeros_like(neighbor_fg_flat), neighbor_fg_flat],
                dim=1,
            )
            _, neighbor_uncertainty_flat, _ = evidential_prob_unc(neighbor_logits)
            neighbor_uncertainty = neighbor_uncertainty_flat.view(
                batch_size, num_neighbors, 1, height, width
            )

            center_logit = foreground_logit.unsqueeze(1)
            center_uncertainty_expanded = center_uncertainty.unsqueeze(1)

            # gamma_{q->p} = sigmoid((u_p - u_q) / tau).
            gate_j2i = torch.sigmoid(
                (center_uncertainty_expanded - neighbor_uncertainty) / (self.tau + 1e-8)
            )
            gate_i2j = torch.sigmoid(
                (neighbor_uncertainty - center_uncertainty_expanded) / (self.tau + 1e-8)
            )

            # phi_{p,q} = tanh(w^T(f_p - f_q)).
            center_feat = aligned_feat.unsqueeze(1)
            neighbor_feat = self.stencil_f(aligned_feat)
            edge_weight = self.edge_mlp(center_feat, neighbor_feat)
            edge_weight = torch.tanh(edge_weight)

            # F_{q->p} = gamma_{q->p} * phi_{p,q} * s_q; F_{p->q} = gamma_{p->q} * s_p.
            flow_in = (gate_j2i * edge_weight * neighbor_foreground_logits).sum(dim=1)
            flow_out = (gate_i2j * center_logit).sum(dim=1)

            # r_p = sigmoid((u_0 - u_p) / tau).
            source_gate = torch.sigmoid((self.u0 - center_uncertainty) / (self.tau + 1e-8))

            # U(s_p) = sum_q(F_{q->p} - F_{p->q}) + r_p * (s_p^0 - s_p^t);
            # s_p^{t+1} = s_p^t + eta * U(s_p^t).
            if self.ugcp_use_source_term:
                foreground_logit = (
                    foreground_logit
                    + self.eta * (flow_in - flow_out)
                    + self.eta * source_gate * (logits[:, 1:2] - foreground_logit)
                )
            else:
                foreground_logit = foreground_logit + self.eta * (flow_in - flow_out)

            refined_logits = torch.cat([refined_logits[:, 0:1], foreground_logit], dim=1)

        return refined_logits


class DepthwiseStencil2D(nn.Module):
    """
    Fixed 2D depthwise stencil that extracts the 4-neighborhood.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * 4,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self._init_weights()
        for p in self.parameters():
            p.requires_grad_(False)

    def _init_weights(self):
        w = torch.zeros((self.channels * 4, 1, 3, 3))
        for c in range(self.channels):
            base = c * 4
            w[base + 0, 0, 1, 2] = 1.0
            w[base + 1, 0, 1, 0] = 1.0
            w[base + 2, 0, 2, 1] = 1.0
            w[base + 3, 0, 0, 1] = 1.0
        self.conv.weight.data.copy_(w)

    def forward(self, x):
        y = self.conv(x)
        batch_size, _, height, width = y.shape
        return y.view(batch_size, 4, self.channels, height, width)


class EdgeMLP2D(nn.Module):
    """
    Edge-wise modulation in decision-aligned feature space.
    """

    def __init__(self, feat_head_channel):
        super().__init__()
        self.linear = nn.Linear(feat_head_channel, 1, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

    def forward(self, center_feat, neighbor_feat):
        feat_diff = center_feat - neighbor_feat
        batch_size, num_neighbors, channels, height, width = feat_diff.shape
        feat_diff = feat_diff.permute(0, 1, 3, 4, 2).contiguous().view(-1, channels)
        edge_weight = self.linear(feat_diff)
        edge_weight = edge_weight.view(
            batch_size, num_neighbors, height, width, 1
        ).permute(0, 1, 4, 2, 3)
        return edge_weight


def evidential_prob_unc(score):
    """
    Return foreground probability, uncertainty, and evidence strength.
    """
    num_classes = score.shape[1]
    evidence = F.softplus(score)
    alpha = evidence + 1.0
    evidence_sum = alpha.sum(dim=1, keepdim=True)
    prob_fg = alpha[:, 1:2] / (evidence_sum + 1e-8)
    unc = num_classes / (evidence_sum + 1e-8)
    return prob_fg, unc, evidence_sum

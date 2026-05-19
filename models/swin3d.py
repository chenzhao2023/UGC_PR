from monai.networks.nets import SwinUNETR
import torch.nn as nn


class SwinUNETR3D(nn.Module):
    """
    Lightweight wrapper around MONAI's SwinUNETR for binary segmentation.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        feature_size=24,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        self.model = SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            patch_size=2,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.1,
            norm_name="instance",
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        return self.model(x)

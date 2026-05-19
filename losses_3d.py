import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from torch.special import digamma


class Seg_loss(nn.Module):
    """
    Segmentation loss for binary logits.
    """

    def __init__(self, weight_dice=1.0, weight_ce=1.0):
        super().__init__()

        self.loss_dicece = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=weight_dice,
            lambda_ce=weight_ce,
        )
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        if torch.any(target > 0):
            target_for_dice = target.unsqueeze(1)
            return self.loss_dicece(logits, target_for_dice)

        return self.loss_ce(logits, target)



class EvidentialSegUQLoss(nn.Module):
    """
    Evidential segmentation loss for binary segmentation.
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_uq: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.lambda_uq = lambda_uq
        self.eps = eps

    def dice_loss(self, prob, target):
        foreground_prob = prob[:, 1]
        foreground_target = (target == 1).float()

        intersection = torch.sum(foreground_prob * foreground_target)
        denom = torch.sum(foreground_prob) + torch.sum(foreground_target)

        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice

    def nll_loss(self, prob, target):
        clamped_prob = torch.clamp(prob, self.eps, 1.0)
        log_prob = torch.log(clamped_prob)

        target_one_hot = F.one_hot(target.long(), num_classes=2)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        nll = -torch.sum(target_one_hot * log_prob, dim=1)
        return nll.mean()

    def uq_loss(self, alpha, target):
        num_classes = alpha.shape[1]
        evidence_sum = alpha.sum(dim=1, keepdim=True)

        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        nll = target_one_hot * (digamma(evidence_sum) - digamma(alpha))
        nll = nll.sum(dim=1).mean()

        alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha
        reg = torch.sum(
            (alpha_tilde - 1)
            * (digamma(alpha_tilde) - digamma(alpha_tilde.sum(dim=1, keepdim=True))),
            dim=1,
        )
        reg = reg.mean()

        return nll + reg

    def forward(self, prob, alpha, target):
        loss_dice = self.dice_loss(prob, target)
        loss_ce = self.nll_loss(prob, target)
        loss_uq = self.uq_loss(alpha, target)

        loss = (
            self.lambda_dice * loss_dice
            + self.lambda_ce * loss_ce
            + self.lambda_uq * loss_uq
        )

        return loss, {
            "dice": loss_dice.detach(),
            "ce": loss_ce.detach(),
            "uq": loss_uq.detach(),
        }

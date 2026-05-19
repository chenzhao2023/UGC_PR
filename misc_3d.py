from models import (
    swin2d,
    swin2d_ugcp,
    swin3d,
    swin3d_ugcp,
    unet2d,
    unet2d_ugcp,
    unet3d,
    unet3d_ugcp,
)
from losses_3d import EvidentialSegUQLoss, Seg_loss


def build_model(cfg):
    model_name = cfg.model.model_name

    if model_name == "Unet2D":
        return unet2d.Unet2D()
    if model_name == "Unet3D":
        return unet3d.Unet3D()
    if model_name == "SwinUNETR2D":
        return swin2d.SwinUNETR2D()
    if model_name == "SwinUNETR3D":
        return swin3d.SwinUNETR3D()
    if model_name == "Unet2D_UGCP":
        return unet2d_ugcp.Unet2D_UGCP(cfg=cfg)
    if model_name == "Unet3D_UGCP":
        return unet3d_ugcp.Unet3D_UGCP(cfg=cfg)
    if model_name == "SwinUNETR2D_UGCP":
        return swin2d_ugcp.SwinUNETR2D_UGCP(cfg=cfg)
    if model_name == "SwinUNETR3D_UGCP":
        return swin3d_ugcp.SwinUNETR3D_UGCP(cfg=cfg)

    raise ValueError(f"Unknown model name: {model_name}")


def get_loss_fn(cfg):
    model_name = cfg.model.model_name

    plain_seg_models = {"Unet2D", "Unet3D", "SwinUNETR2D", "SwinUNETR3D"}
    ugcp_models = {
        "Unet2D_UGCP",
        "Unet3D_UGCP",
        "SwinUNETR2D_UGCP",
        "SwinUNETR3D_UGCP",
    }

    if model_name in plain_seg_models:
        return Seg_loss(weight_dice=cfg.model.weight_dice, weight_ce=cfg.model.weight_ce)

    if model_name in ugcp_models:
        return EvidentialSegUQLoss(
            cfg.model.weight_dice,
            cfg.model.weight_ce,
            cfg.model.weight_uq,
        )

    raise ValueError(f"Unknown model name: {model_name}")

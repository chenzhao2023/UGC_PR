import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric
from tqdm import tqdm


PLAIN_MODEL_NAMES = {"Unet2D", "Unet3D", "SwinUNETR2D", "SwinUNETR3D"}


def train_one_epoch(
    cfg,
    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
    scheduler=None,
    warmup_scheduler=None,
):
    model.train()

    epoch_loss = 0.0
    epoch_dice = 0.0
    amp_dtype = torch.float16 if cfg.amp.amp_dtype == "fp16" else torch.bfloat16

    data_iter = tqdm(
        train_loader,
        desc=f"Train Epoch {epoch}",
        leave=True,
        total=len(train_loader),
    )

    for batch in data_iter:
        image = batch["img"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            enabled=cfg.amp.use_amp,
            dtype=amp_dtype,
            device_type=device,
        ):
            logits = model(image)
            if cfg.model.model_name in PLAIN_MODEL_NAMES:
                loss = criterion(logits, label)
            else:
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                prob = alpha / alpha.sum(dim=1, keepdim=True)
                loss, _ = criterion(prob, alpha, label)

        pred_fg = torch.argmax(logits, dim=1) == 1
        gt_fg = label == 1

        intersection = torch.sum(pred_fg & gt_fg)
        cardinality = torch.sum(pred_fg) + torch.sum(gt_fg)
        batch_dice = (2.0 * intersection + 1e-8) / (cardinality + 1e-8)

        epoch_dice += batch_dice.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    if warmup_scheduler is not None:
        warmup_scheduler.step()

    if scheduler is not None:
        scheduler.step(epoch + 1)

    return {
        "loss": epoch_loss / len(train_loader),
        "dice": epoch_dice / len(train_loader),
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def validate(cfg, model, val_loader, device):
    model.eval()

    if cfg.runtime.compute_hd95:
        hd95_metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction="none",
        )
    else:
        hd95_metric = None

    tp = tn = fp = fn = 0.0
    dice_sum = 0.0
    num_cases = 0
    hd95_list = []

    data_iter = tqdm(val_loader, desc="Validate", leave=True)

    for batch in data_iter:
        image = batch["img"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        logits = sliding_window_inference(
            image,
            roi_size=cfg.data.patch_size,
            sw_batch_size=8,
            predictor=model,
            overlap=cfg.data.overlap_ratio,
            mode="gaussian",
        )

        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        prob = alpha / alpha.sum(dim=1, keepdim=True)

        pred = torch.argmax(prob, dim=1)

        pred_bin = pred == 1
        gt_bin = label == 1

        intersection = torch.sum(pred_bin & gt_bin)
        cardinality = torch.sum(pred_bin) + torch.sum(gt_bin)
        dice = (2.0 * intersection + 1e-8) / (cardinality + 1e-8)
        dice_sum += dice.item()
        num_cases += 1

        if hd95_metric is not None:
            hd95_metric(pred_bin.unsqueeze(1).float(), gt_bin.unsqueeze(1).float())
            hd95_val = hd95_metric.get_buffer()[-1].item()
            hd95_list.append(hd95_val)
            hd95_metric.reset()

        tp += torch.sum(pred_bin & gt_bin).item()
        tn += torch.sum(~pred_bin & ~gt_bin).item()
        fp += torch.sum(pred_bin & ~gt_bin).item()
        fn += torch.sum(~pred_bin & gt_bin).item()

    dice = dice_sum / max(num_cases, 1)

    if hd95_metric is not None and hd95_list:
        hd95 = sum(hd95_list) / len(hd95_list)
    else:
        hd95 = 0.0

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "dice": dice,
        "hd95": hd95,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric
from skimage.morphology import skeletonize_3d
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs_3d import parse_args
from dataset_3d import MRACaseDataset, load_case_list
from misc_3d import build_model


PROJECT_DIR = Path(__file__).resolve().parent


def compute_dice(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    cardinality = pred_mask.sum() + gt_mask.sum()
    return float((2.0 * intersection + 1e-8) / (cardinality + 1e-8))


def compute_cldice(pred_mask, gt_mask):
    pred_np = pred_mask.cpu().numpy().astype(bool)
    gt_np = gt_mask.cpu().numpy().astype(bool)

    if pred_np.sum() < 10 or gt_np.sum() < 10:
        return 0.0

    pred_skeleton = skeletonize_3d(pred_np)
    gt_skeleton = skeletonize_3d(gt_np)

    topology_precision = (pred_skeleton & gt_np).sum() / (pred_skeleton.sum() + 1e-8)
    topology_sensitivity = (gt_skeleton & pred_np).sum() / (gt_skeleton.sum() + 1e-8)

    return float(
        (2 * topology_precision * topology_sensitivity)
        / (topology_precision + topology_sensitivity + 1e-8)
    )


def compute_hd95(pred_mask, gt_mask, metric):
    metric(
        pred_mask.unsqueeze(0).unsqueeze(0).float(),
        gt_mask.unsqueeze(0).unsqueeze(0).float(),
    )
    metric_buffer = metric.get_buffer()
    metric_value = float(metric_buffer[-1]) if len(metric_buffer) > 0 else 0.0
    metric.reset()
    return metric_value


def decode_logits(logits):
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)

    evidence = F.softplus(logits)
    alpha = evidence + 1
    return alpha / alpha.sum(dim=1, keepdim=True)


def parse_experiment_name(experiment_name):
    parts = experiment_name.split("_")
    for index, part in enumerate(parts):
        if part.startswith("fold"):
            return "_".join(parts[:index]), int(part.replace("fold", ""))
    raise ValueError(f"Cannot parse fold from experiment name: {experiment_name}")


def collect_experiments(experiments_dir):
    experiment_paths = []
    for experiment_path in sorted(experiments_dir.iterdir()):
        if not experiment_path.is_dir():
            continue
        if not (experiment_path / "best.pth").exists():
            continue
        experiment_paths.append(experiment_path)
    return experiment_paths


def split_experiment_name(experiment_name):
    model_name, fold = parse_experiment_name(experiment_name)
    fold_token = f"fold{fold}"
    parts = experiment_name.split("_")
    fold_index = parts.index(fold_token)
    timestamp = "_".join(parts[fold_index + 1:]) if fold_index + 1 < len(parts) else ""
    return model_name, fold, timestamp


def build_val_loader(cfg):
    val_dataset = MRACaseDataset(
        case_list=load_case_list(cfg.data.val_txt),
        root_images=cfg.data.root_images,
        root_labels=cfg.data.root_labels,
    )
    return DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)


@torch.no_grad()
def evaluate_experiment(cfg, model, loader, device, prediction_dir, csv_path):
    model.eval()
    prediction_dir.mkdir(parents=True, exist_ok=True)

    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="none",
    )

    records = []

    for batch in tqdm(loader, desc=f"{cfg.model.model_name}-fold{cfg.data.fold}", leave=False):
        image = batch["img"].to(device)
        gt_mask = batch["label"].to(device)
        case_name = batch["case_id"][0]

        logits = sliding_window_inference(
            inputs=image,
            roi_size=cfg.data.patch_size,
            sw_batch_size=4,
            predictor=model,
            overlap=cfg.data.overlap_ratio,
            mode="gaussian",
        )
        prob = decode_logits(logits)

        pred_mask = torch.argmax(prob, dim=1)[0] == 1
        gt_mask = gt_mask[0] == 1

        dice = compute_dice(pred_mask, gt_mask)
        cldice = compute_cldice(pred_mask, gt_mask)
        hd95 = compute_hd95(pred_mask, gt_mask, hd95_metric)

        records.append(
            {
                "case": case_name,
                "dice": dice,
                "cldice": cldice,
                "hd95": hd95,
            }
        )

        pred_path = prediction_dir / f"{Path(case_name).stem}.npz"
        np.savez_compressed(pred_path, pred=pred_mask.cpu().numpy().astype(np.bool_))

    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["case", "dice", "cldice", "hd95"])
        writer.writeheader()
        writer.writerows(records)

    mean_metrics = {
        "dice": sum(record["dice"] for record in records) / len(records),
        "cldice": sum(record["cldice"] for record in records) / len(records),
        "hd95": sum(record["hd95"] for record in records) / len(records),
    }
    return mean_metrics


def save_summary_csv(summary_path, summary_rows):
    fieldnames = [
        "experiment_name",
        "model_name",
        "fold",
        "timestamp",
        "dice",
        "cldice",
        "hd95",
    ]
    with summary_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    cfg = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiments_dir = PROJECT_DIR / "experiments"
    results_dir = PROJECT_DIR / "results"
    predictions_root = PROJECT_DIR / "pred_3d"

    results_dir.mkdir(parents=True, exist_ok=True)
    experiment_paths = collect_experiments(experiments_dir)
    summary_rows = []

    for experiment_path in experiment_paths:

        model_name, fold = parse_experiment_name(experiment_path.name)
        cfg.model.model_name = model_name
        cfg.data.fold = fold

        print(f"\n[TEST ] {experiment_path.name}")

        model = build_model(cfg).to(device)
        checkpoint = torch.load(experiment_path / "best.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=True)

        val_loader = build_val_loader(cfg)
        exp_name = experiment_path.name
        csv_path = results_dir / f"results_{exp_name}.csv"
        prediction_dir = predictions_root / exp_name

        metrics = evaluate_experiment(cfg, model, val_loader, device, prediction_dir, csv_path)
        _, _, timestamp = split_experiment_name(experiment_path.name)
        summary_rows.append(
            {
                "experiment_name": experiment_path.name,
                "model_name": model_name,
                "fold": fold,
                "timestamp": timestamp,
                "dice": metrics["dice"],
                "cldice": metrics["cldice"],
                "hd95": metrics["hd95"],
            }
        )
        print(
            f"[DONE ] {experiment_path.name} | "
            f"Dice={metrics['dice']:.4f} | "
            f"clDice={metrics['cldice']:.4f} | "
            f"HD95={metrics['hd95']:.2f}"
        )

    summary_rows.sort(key=lambda row: (row["model_name"], row["fold"], row["timestamp"]))
    summary_path = results_dir / "summary_all_experiments.csv"
    save_summary_csv(summary_path, summary_rows)
    print(f"\n[SUMMARY] Saved: {summary_path}")


if __name__ == "__main__":
    main()

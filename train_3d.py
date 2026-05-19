import logging
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs_3d import parse_args
from dataset_3d import MRACaseDataset, OfflineMRAPatchDataset, load_case_list
from misc_3d import build_model, get_loss_fn
from one_epoch_3d import train_one_epoch, validate


warnings.filterwarnings("ignore", category=UserWarning)


def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def setup_logger(log_dir: str, name="train"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    return logger


def main():
    cfg = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = cfg.system.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.system.cudnn_deterministic
    torch.backends.cuda.matmul.allow_tf32 = cfg.system.tf32
    torch.set_float32_matmul_precision(cfg.system.matmul_precision)

    os.makedirs(cfg.runtime.save_dir, exist_ok=True)
    os.makedirs(cfg.runtime.logging_dir, exist_ok=True)

    logger = setup_logger(cfg.runtime.logging_dir)
    logger.info("========== Training started ==========")
    logger.info(f"Save dir: {cfg.runtime.save_dir}")

    logger.info("Config:")
    for group_cfg in vars(cfg).values():
        for key, value in vars(group_cfg).items():
            logger.info(f"  {key}: {value}")

    writer = SummaryWriter(cfg.runtime.logging_dir)
    model = build_model(cfg).to(device)

    train_dataset = OfflineMRAPatchDataset(
        case_list=load_case_list(cfg.data.train_txt),
        root_images=cfg.data.patch_images,
        root_labels=cfg.data.patch_labels,
        samples_per_case=cfg.data.samples_per_case,
    )

    val_dataset = MRACaseDataset(
        case_list=load_case_list(cfg.data.val_txt),
        root_images=cfg.data.root_images,
        root_labels=cfg.data.root_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        persistent_workers=cfg.system.persistent_workers,
        prefetch_factor=cfg.system.prefetch_factor,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=cfg.system.pin_memory,
        persistent_workers=cfg.system.persistent_workers,
        prefetch_factor=cfg.system.prefetch_factor,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr.base_lr,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.lr.T_0,
        T_mult=cfg.lr.T_mult,
        eta_min=cfg.lr.eta_min,
    )

    if cfg.lr.use_warmup:
        steps_per_epoch = len(train_loader)
        warmup_steps = cfg.lr.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step >= warmup_steps:
                return 1.0
            return float(step + 1) / float(warmup_steps)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
    else:
        warmup_scheduler = None

    scaler = torch.amp.GradScaler(
        device=device,
        enabled=cfg.amp.use_amp and cfg.amp.amp_dtype == "fp16",
        init_scale=cfg.amp.init_scale,
        growth_interval=cfg.amp.growth_interval,
    )

    criterion = get_loss_fn(cfg)
    best_metric = -1e9
    patience = 0

    for epoch in range(cfg.train.epoch):
        train_dataset.set_epoch(epoch)

        train_stats = train_one_epoch(
            cfg,
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
            scheduler,
            warmup_scheduler,
        )

        logger.info(
            f"[TRAIN] Epoch {epoch:03d}/{cfg.train.epoch:03d} | "
            f'Dice={train_stats["dice"]:.4f} | '
            f"loss={train_stats['loss']:.4f} | "
            f"lr={train_stats['lr']:.4e}"
        )

        writer.add_scalar("train/loss", train_stats["loss"], epoch)
        writer.add_scalar("train/lr", train_stats["lr"], epoch)

        state_dict = model.state_dict() if cfg.ddp.use_ddp else model.state_dict()

        if epoch % cfg.train.valid_freq == 0:
            metrics = validate(cfg, model, val_loader, device)

            logger.info(
                f"[VAL  ] Epoch {epoch:03d}/{cfg.train.epoch:03d} | "
                f"Dice={metrics['dice']:.4f} | "
                f"HD95={metrics['hd95']:.2f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Spec={metrics['specificity']:.4f}"
            )

            for key, value in metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)

            if metrics["dice"] > best_metric:
                best_metric = metrics["dice"]
                patience = 0
                torch.save(state_dict, os.path.join(cfg.runtime.save_dir, "best.pth"))
                logger.info(
                    f"[CKPT ] Epoch {epoch:03d}/{cfg.train.epoch:03d} | "
                    f"New BEST Dice={best_metric:.4f} | saved=best.pth"
                )
            else:
                patience += 1
                logger.info(
                    f"[STAT ] Epoch {epoch:03d}/{cfg.train.epoch:03d} | "
                    f"Best Dice={best_metric:.4f} | "
                    f"Patience {patience}/{cfg.train.earlystop}"
                )

            if patience >= cfg.train.earlystop:
                logger.info(
                    f"[STOP ] Epoch {epoch:03d}/{cfg.train.epoch:03d} | "
                    f"Early stopping triggered | "
                    f"Best Dice={best_metric:.4f}"
                )
                break


if __name__ == "__main__":
    main()

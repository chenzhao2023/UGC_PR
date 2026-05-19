import argparse
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent


class ModelConfig:
    SUPPORTED_MODEL_NAMES = (
        "Unet2D",
        "Unet3D",
        "SwinUNETR2D",
        "SwinUNETR3D",
        "Unet2D_UGCP",
        "Unet3D_UGCP",
        "SwinUNETR2D_UGCP",
        "SwinUNETR3D_UGCP",
    )

    def __init__(self):
        self.model_name = "Unet3D"
        self.weight_dice = 0.4
        self.weight_ce = 0.4
        self.weight_uq = 0.2

        self.use_ugcp = True
        self.ugcp_steps = 2
        self.ugcp_use_source_term = True
        self.ugcp_eta = 1.0
        self.ugcp_u0 = 0.5
        self.ugcp_tau = 0.1


class TrainConfig:
    def __init__(self):
        self.epoch = 800
        self.batch_size = 8
        self.weight_decay = 1e-4
        self.seed = 42
        self.earlystop = 6
        self.valid_freq = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class DataConfig:
    def __init__(self):
        self.fold = 4
        self.mini_data = False
        self.patch_images = str(PROJECT_DIR / "Dataset" / "Preprocessed" / "patches_images")
        self.patch_labels = str(PROJECT_DIR / "Dataset" / "Preprocessed" / "patches_labels")
        self.root_images = str(PROJECT_DIR / "Dataset" / "Preprocessed" / "images")
        self.root_labels = str(PROJECT_DIR / "Dataset" / "Preprocessed" / "labels")
        self.train_txt = ""
        self.val_txt = ""
        self.patch_size = (64, 64, 64)
        self.samples_per_case = 32
        self.overlap_ratio = 0.5


class LRConfig:
    def __init__(self):
        self.base_lr = 1e-3
        self.use_warmup = False
        self.warmup_epochs = 5
        self.warmup_start_lr = 1e-5
        self.scheduler = "cosine"
        self.T_0 = 60
        self.T_mult = 2
        self.eta_min = 1e-6


class AMPConfig:
    def __init__(self):
        self.use_amp = False
        self.amp_dtype = "fp16"
        self.init_scale = 1024
        self.growth_interval = 2000


class DDPConfig:
    def __init__(self):
        self.use_ddp = False
        self.backend = "nccl"
        self.init_method = "env://"
        self.sync_bn = False
        self.find_unused_parameters = False


class SystemConfig:
    def __init__(self):
        self.cudnn_benchmark = True
        self.cudnn_deterministic = False
        self.tf32 = True
        self.matmul_precision = "high"
        self.num_workers = 4
        self.persistent_workers = False
        self.prefetch_factor = 8
        self.pin_memory = True


class RuntimeConfig:
    def __init__(self):
        self.time_stamp = ""
        self.save_dir = ""
        self.logging_dir = ""
        self.check_point = False
        self.check_point_path = ""
        self.compute_hd95 = False


class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.train = TrainConfig()
        self.lr = LRConfig()
        self.amp = AMPConfig()
        self.ddp = DDPConfig()
        self.system = SystemConfig()
        self.runtime = RuntimeConfig()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)
    parser.add_argument("--mini_data", action="store_true", default=None)
    parser.add_argument("--patch_images", type=str)
    parser.add_argument("--patch_labels", type=str)
    parser.add_argument("--root_images", type=str)
    parser.add_argument("--root_labels", type=str)
    parser.add_argument("--train_txt", type=str)
    parser.add_argument("--val_txt", type=str)

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--weight_dice", type=float)
    parser.add_argument("--weight_ce", type=float)
    parser.add_argument("--weight_uq", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--use_ugcp", action="store_true", default=None)
    parser.add_argument("--ugcp_steps", type=int)
    parser.add_argument("--ugcp_use_source_term", action="store_true", default=None)
    parser.add_argument("--ugcp_eta", type=float)
    parser.add_argument("--ugcp_u0", type=float)
    parser.add_argument("--ugcp_tau", type=float)

    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--earlystop", type=int)
    parser.add_argument("--valid_freq", type=int)

    parser.add_argument("--base_lr", type=float)
    parser.add_argument("--use_warmup", action="store_true", default=None)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--warmup_start_lr", type=float)
    parser.add_argument("--scheduler", type=str, choices=["cosine", "step", "plateau"])
    parser.add_argument("--T_0", type=int)
    parser.add_argument("--T_mult", type=int)
    parser.add_argument("--eta_min", type=float)

    parser.add_argument("--use_amp", action="store_true", default=None)
    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16"])

    parser.add_argument("--use_ddp", action="store_true", default=None)
    parser.add_argument("--sync_bn", action="store_true", default=None)
    parser.add_argument("--find_unused_parameters", action="store_true", default=None)

    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--prefetch_factor", type=int)
    parser.add_argument("--pin_memory", action="store_true", default=None)
    parser.add_argument("--persistent_workers", action="store_true", default=None)
    parser.add_argument("--tf32", action="store_true", default=None)
    parser.add_argument("--matmul_precision", type=str, choices=["high", "medium"])

    parser.add_argument("--check_point", action="store_true", default=None)
    parser.add_argument("--check_point_path", type=str)
    parser.add_argument("--note", type=str)
    parser.add_argument("--compute_hd95", action="store_true", default=None)

    args = parser.parse_args()
    cfg = Config()

    for key, value in vars(args).items():
        if value is None:
            continue
        for section in vars(cfg).values():
            if hasattr(section, key):
                setattr(section, key, value)

    cfg.runtime.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg.runtime.save_dir = (
        f"experiments/{cfg.model.model_name}"
        f"_fold{cfg.data.fold}"
        f"_{cfg.runtime.time_stamp}"
    )
    cfg.runtime.logging_dir = f"{cfg.runtime.save_dir}/logs"

    split_name = "split_10per" if cfg.data.mini_data else "split"
    split_dir = PROJECT_DIR / "Dataset" / "Preprocessed" / split_name
    if cfg.data.train_txt == "":
        cfg.data.train_txt = str(split_dir / f"fold{cfg.data.fold}_train.txt")
    if cfg.data.val_txt == "":
        cfg.data.val_txt = str(split_dir / f"fold{cfg.data.fold}_val.txt")

    return cfg

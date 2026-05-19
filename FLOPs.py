import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR
REPO_ROOT = PROJECT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS = {
    "3d": {
        "project_dir": PROJECT_DIR,
        "config_module": "configs_3d",
        "misc_module": "misc_3d",
        "default_models": ("Unet3D", "Unet3D_UGCP", "SwinUNETR3D", "SwinUNETR3D_UGCP"),
        "default_input_shape": (1, 96, 96, 96),
    },
    "2d": {
        "project_dir": PROJECT_DIR,
        "config_module": "configs_3d",
        "misc_module": "misc_3d",
        "default_models": ("Unet2D", "Unet2D_UGCP", "SwinUNETR2D", "SwinUNETR2D_UGCP"),
        "default_input_shape": (1, 512, 512),
    },
}

REPORT_GROUPS = (
    {
        "name": "2D UNet",
        "task": "2d",
        "models": ("Unet2D", "Unet2D_UGCP"),
    },
    {
        "name": "2D SwinUNETR",
        "task": "2d",
        "models": ("SwinUNETR2D", "SwinUNETR2D_UGCP"),
    },
    {
        "name": "3D UNet",
        "task": "3d",
        "models": ("Unet3D", "Unet3D_UGCP"),
    },
    {
        "name": "3D SwinUNETR",
        "task": "3d",
        "models": ("SwinUNETR3D", "SwinUNETR3D_UGCP"),
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare model complexity and inference time for 2D/3D segmentation models."
    )
    parser.add_argument("--mode", choices=["all", "2d", "3d"], default="all")
    parser.add_argument("--task", choices=sorted(TASKS), default=None, help="Manual benchmark task.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to benchmark. Defaults depend on --task.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Input tensor shape excluding batch dimension.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Optional CSV output path.",
    )
    args = parser.parse_args()

    if args.task is None and args.mode != "all":
        args.task = args.mode

    if args.task is not None:
        task_cfg = TASKS[args.task]
        if args.models is None:
            args.models = list(task_cfg["default_models"])
        if args.input_shape is None:
            args.input_shape = list(task_cfg["default_input_shape"])

    if args.task is not None:
        expected_dims = len(TASKS[args.task]["default_input_shape"])
    elif args.input_shape is not None:
        raise ValueError("--input_shape requires --task 2d or --task 3d")
    else:
        expected_dims = None

    if expected_dims is not None and len(args.input_shape) != expected_dims:
        raise ValueError(
            f"{args.task} expects {expected_dims} input dims excluding batch, "
            f"got {len(args.input_shape)}: {args.input_shape}"
        )

    return args


def load_task_api(task: str):
    import importlib

    task_cfg = TASKS[task]
    project_dir = task_cfg["project_dir"]
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    config_module = importlib.import_module(task_cfg["config_module"])
    misc_module = importlib.import_module(task_cfg["misc_module"])
    return config_module.Config, misc_module.build_model


def make_cfg(task: str, model_name: str):
    Config, _ = load_task_api(task)
    cfg = Config()
    cfg.model.model_name = model_name
    return cfg


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_count(value: float):
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:.3f}G"
    if abs_value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"


def parse_count_string(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().upper()
    tokens = text.split()
    if len(tokens) > 0:
        text = tokens[0]

    unit_scale = 1.0
    if text.endswith("TFLOPS") or text.endswith("TMACS") or text.endswith("TB"):
        text = text[:-6] if text.endswith("TFLOPS") else text
    suffix_map = {
        "T": 1e12,
        "G": 1e9,
        "M": 1e6,
        "K": 1e3,
        "B": 1e9,
    }
    if text and text[-1] in suffix_map:
        unit_scale = suffix_map[text[-1]]
        text = text[:-1]

    text = text.replace(",", "")
    try:
        return float(text) * unit_scale
    except ValueError:
        return None


def try_calflops_profile(model: nn.Module, example_input: torch.Tensor):
    try:
        from calflops import calculate_flops  # type: ignore
    except Exception:
        return None, None, None, "unavailable"

    try:
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=tuple(example_input.shape),
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
        return (
            parse_count_string(flops),
            parse_count_string(macs),
            parse_count_string(params),
            "calflops",
        )
    except Exception as exc:
        return None, None, None, f"calflops_failed: {exc}"


def try_thop_profile(model: nn.Module, example_input: torch.Tensor):
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None, None, None, "unavailable"

    try:
        macs, params = profile(model, inputs=(example_input,), verbose=False)
        flops = 2.0 * float(macs)
        return flops, float(macs), float(params), "thop"
    except Exception as exc:
        return None, None, None, f"thop_failed: {exc}"


class ManualFlopCounter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.total_flops = 0

    def _conv_hook(self, module: nn.Module, inputs, output):
        x = inputs[0]
        if not isinstance(output, torch.Tensor):
            return

        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_spatial = output.shape[2:]

        kernel_ops = 1
        for k in module.kernel_size:
            kernel_ops *= k

        in_channels_per_group = module.in_channels // module.groups
        output_elements = batch_size * out_channels
        for dim in out_spatial:
            output_elements *= dim

        mul_add = 2 * in_channels_per_group * kernel_ops
        bias_ops = 1 if module.bias is not None else 0
        self.total_flops += output_elements * (mul_add + bias_ops)

    def _linear_hook(self, module: nn.Module, inputs, output):
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return

        batch_elements = 1
        for dim in x.shape[:-1]:
            batch_elements *= dim

        mul_add = 2 * module.in_features
        bias_ops = 1 if module.bias is not None else 0
        self.total_flops += batch_elements * module.out_features * (mul_add + bias_ops)

    def _register(self):
        for module in self.model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                self.handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))

    def _remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def profile(self, example_input: torch.Tensor):
        self.total_flops = 0
        self._register()
        try:
            with torch.inference_mode():
                _ = self.model(example_input)
        finally:
            self._remove()
        return float(self.total_flops)


def estimate_flops(model: nn.Module, example_input: torch.Tensor):
    flops, macs, params_from_calflops, source = try_calflops_profile(model, example_input)
    if flops is not None:
        return flops, macs, params_from_calflops, source

    flops, macs, params_from_thop, source = try_thop_profile(model, example_input)
    if flops is not None:
        return flops, macs, params_from_thop, source

    counter = ManualFlopCounter(model)
    flops = counter.profile(example_input)
    return flops, None, None, "manual_conv_linear_only"


def synchronize(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_inference_time(
    model: nn.Module,
    example_input: torch.Tensor,
    device: str,
    warmup: int,
    iters: int,
    use_amp: bool,
):
    amp_enabled = use_amp and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"

    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
                _ = model(example_input)
        synchronize(device)

        if device == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(iters):
                with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
                    _ = model(example_input)
            ender.record()
            torch.cuda.synchronize()
            total_ms = starter.elapsed_time(ender)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                with torch.autocast(device_type=autocast_device, enabled=False):
                    _ = model(example_input)
            total_ms = (time.perf_counter() - t0) * 1000.0

    mean_ms = total_ms / max(iters, 1)
    throughput = example_input.shape[0] * 1000.0 / mean_ms
    return mean_ms, throughput


def maybe_set_threads(device: str, num_threads: int):
    if device == "cpu" and num_threads > 0:
        torch.set_num_threads(num_threads)


def build_example_input(batch_size: int, input_shape, device: str):
    shape = (batch_size, *input_shape)
    return torch.randn(shape, device=device)


def benchmark_one_model(args, task: str, model_name: str, input_shape):
    cfg = make_cfg(task, model_name)
    _, build_model = load_task_api(task)
    model = build_model(cfg).to(args.device).eval()
    example_input = build_example_input(args.batch_size, tuple(input_shape), args.device)

    total_params, trainable_params = count_parameters(model)
    flops, macs, params_from_profiler, flop_source = estimate_flops(model, example_input)
    mean_ms, throughput = benchmark_inference_time(
        model=model,
        example_input=example_input,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        use_amp=args.use_amp,
    )

    return {
        "model": model_name,
        "task": task,
        "input_shape": tuple(input_shape),
        "params": total_params,
        "trainable_params": trainable_params,
        "params_from_thop": params_from_profiler,
        "flops": flops,
        "macs": macs,
        "gflops": flops / 1e9,
        "gmacs": (macs / 1e9) if macs is not None else None,
        "mean_ms": mean_ms,
        "throughput": throughput,
        "flop_source": flop_source,
    }


def print_group_results(group_name, task, input_shape, results, args):
    print("")
    print(f"Benchmark: {group_name}")
    print(f"  task        : {task}")
    print(f"  device      : {args.device}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  input_shape : {(args.batch_size, *input_shape)}")
    print(f"  warmup      : {args.warmup}")
    print(f"  iters       : {args.iters}")
    print(f"  amp         : {args.use_amp}")
    print("")

    header = (
        f"{'Model':<16}"
        f"{'Params':>14}"
        f"{'Trainable':>14}"
        f"{'GFLOPs':>14}"
        f"{'GMACs':>14}"
        f"{'Mean Inference time ms':>12}"
        f"{'Samples/s':>12}"
        f"{'FLOPs source':>24}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['model']:<16}"
            f"{format_count(row['params']):>14}"
            f"{format_count(row['trainable_params']):>14}"
            f"{row['gflops']:>14.3f}"
            f"{(row['gmacs'] if row['gmacs'] is not None else float('nan')):>14.3f}"
            f"{row['mean_ms']:>12.3f}"
            f"{row['throughput']:>12.3f}"
            f"{row['flop_source']:>24}"
        )


def save_csv(csv_path: str, results):
    import csv

    fieldnames = [
        "model",
        "task",
        "input_shape",
        "params",
        "trainable_params",
        "params_from_thop",
        "flops",
        "macs",
        "gflops",
        "gmacs",
        "mean_ms",
        "throughput",
        "flop_source",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def selected_groups(args):
    if args.task is not None:
        return (
            {
                "name": f"{args.task} custom",
                "task": args.task,
                "models": tuple(args.models),
                "input_shape": tuple(args.input_shape),
            },
        )

    groups = REPORT_GROUPS
    if args.mode == "2d":
        groups = tuple(group for group in groups if group["task"] == "2d")
    elif args.mode == "3d":
        groups = tuple(group for group in groups if group["task"] == "3d")

    normalized = []
    for group in groups:
        normalized.append(
            {
                **group,
                "input_shape": TASKS[group["task"]]["default_input_shape"],
            }
        )
    return tuple(normalized)


def main():
    args = parse_args()
    maybe_set_threads(args.device, args.num_threads)

    all_results = []
    for group in selected_groups(args):
        group_results = []
        for model_name in group["models"]:
            print(f"[RUN ] {group['name']} | {model_name}")
            result = benchmark_one_model(args, group["task"], model_name, group["input_shape"])
            group_results.append(result)
            all_results.append(result)
            print(
                f"[DONE] {model_name} | "
                f"Params={format_count(result['params'])} | "
                f"GFLOPs={result['gflops']:.3f} | "
                f"Mean={result['mean_ms']:.3f} ms"
            )
        print_group_results(group["name"], group["task"], group["input_shape"], group_results, args)

    if args.save_csv:
        save_csv(args.save_csv, all_results)
        print(f"\nSaved CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()

# Model                   Params     Trainable        GFLOPs           Inference time ms   Samples/s            FLOPs source
# ----------------------------------------------------------------------------------------------------------------------------------
# Unet3D                  4.806M        4.806M        54.498             5.903              169.399                    thop
# Unet3D_UGCP             4.807M        4.806M        58.709            18.056              55.384                    thop

# Model                   Params     Trainable        GFLOPs          Inference time ms    Samples/s            FLOPs source
# ----------------------------------------------------------------------------------------------------------------------------------
# SwinUNETR3D            15.703M       15.703M       399.457          123.900                8.071                    thop
# SwinUNETR3D_UGCP       15.704M       15.703M       403.668          138.939                7.197                    thop

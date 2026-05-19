import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = DATASET_DIR / "Raw" 
DEFAULT_SAVE_ROOT = DATASET_DIR / "Preprocessed"


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess 3D NIfTI volumes to resampled NPY files.")
    parser.add_argument("--raw_root", type=str, default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--save_root", type=str, default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--target_spacing", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--max_workers", type=int, default=16)
    return parser.parse_args()


def normalize(image):
    p1, p99 = np.percentile(image, (1, 99))
    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1 + 1e-8)
    return image.astype(np.float32)


def resample(image, spacing, new_spacing, is_label=False):
    resize_factor = np.array(spacing) / np.array(new_spacing)
    new_shape = np.round(image.shape * resize_factor).astype(int)
    zoom_factor = new_shape / image.shape
    interpolation_order = 0 if is_label else 1
    return zoom(image, zoom_factor, order=interpolation_order)


def process_case(task):
    in_path, out_path, is_label, target_spacing = task

    try:
        nii = nib.load(in_path)

        data = nii.get_fdata()
        spacing = nii.header.get_zooms()[:3]
        affine = nii.affine

        data = resample(data, spacing, target_spacing, is_label)

        if is_label:
            data = (data > 0).astype(np.uint8)
        else:
            data = normalize(data)

        np.save(out_path, data)

        if not is_label:
            meta = {
                "original_spacing": [float(value) for value in spacing],
                "target_spacing": [float(value) for value in target_spacing],
                "original_shape": [int(value) for value in nii.shape],
                "resampled_shape": [int(value) for value in data.shape],
                "affine": affine.tolist(),
            }

            meta_path = out_path.replace(".npy", ".json")
            with open(meta_path, "w") as file_obj:
                json.dump(meta, file_obj, indent=4)

        return data.shape

    except Exception as exc:
        print(f"Error: {in_path}")
        print(exc)
        return None


def build_tasks(raw_root, save_root, split, is_label):
    tasks = []

    raw_folder_name = "labels" if is_label else "images"
    output_prefix = "labels" if is_label else "images"
    raw_dir = os.path.join(raw_root, f"{raw_folder_name}{split}")
    save_dir = os.path.join(save_root, "labels" if is_label else "images")

    if not os.path.exists(raw_dir):
        return tasks

    for root_dir, _, files in os.walk(raw_dir):
        for file_name in files:
            if not file_name.endswith(".nii.gz"):
                continue

            in_path = os.path.join(root_dir, file_name)

            case_name = file_name.replace(".nii.gz", "")
            prefix = f"{output_prefix}{split}"
            out_name = f"{prefix}_{case_name}.npy"
            out_path = os.path.join(save_dir, out_name)

            tasks.append((in_path, out_path, is_label))

    return tasks


def run_parallel(tasks, max_workers):
    shapes = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_case, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                shapes.append(result)

    return shapes


if __name__ == "__main__":
    args = parse_args()
    raw_root = args.raw_root
    save_root = args.save_root
    target_spacing = tuple(args.target_spacing)
    max_workers = args.max_workers

    os.makedirs(os.path.join(save_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "labels"), exist_ok=True)

    print("Building tasks...")

    tasks = []
    tasks += build_tasks(raw_root, save_root, "Tr", is_label=False)
    tasks += build_tasks(raw_root, save_root, "Tr", is_label=True)
    tasks += build_tasks(raw_root, save_root, "Ts", is_label=False)
    tasks += build_tasks(raw_root, save_root, "Ts", is_label=True)
    tasks = [(in_path, out_path, is_label, target_spacing) for in_path, out_path, is_label in tasks]

    print(f"Total tasks: {len(tasks)}")
    print("\nStart preprocessing...")

    shapes = run_parallel(tasks, max_workers)
    print("\n===== Resampled Shape Statistics =====")

    shapes = [shape for shape in shapes if shape is not None]

    if len(shapes) == 0:
        print("No valid data processed")
        raise SystemExit(1)

    z_axis_sizes = [shape[2] for shape in shapes]
    z_axis_array = np.array(z_axis_sizes)

    print(f"Num samples: {len(shapes)}")
    print(f"min z: {z_axis_array.min()}")
    print(f"max z: {z_axis_array.max()}")
    print(f"mean z: {z_axis_array.mean():.2f}")

    print("\nSuggested patch depth:")
    print(f"safe value: {int(z_axis_array.min())}")
    print(
        f"recommended range: {int(np.percentile(z_axis_array, 20))} ~ "
        f"{int(np.percentile(z_axis_array, 50))}"
    )

    print("\nDone.")

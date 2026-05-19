import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = DATASET_DIR / "Preprocessed"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline 3D training patches from preprocessed volumes.")
    parser.add_argument("--preprocessed_dir", type=str, default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--patch_size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--patch_per_case", type=int, default=128)
    parser.add_argument("--uniform_ratio", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_npy(path):
    return np.load(path)


def get_valid_centers(shape, patch_size):
    depth, height, width = shape
    patch_depth, patch_height, patch_width = patch_size

    margin_depth = patch_depth // 2
    margin_height = patch_height // 2
    margin_width = patch_width // 2

    z_coords = np.arange(margin_depth, depth - margin_depth)
    y_coords = np.arange(margin_height, height - margin_height)
    x_coords = np.arange(margin_width, width - margin_width)

    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    centers = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    return centers


def get_uniform_centers(shape, patch_size, num_samples):
    depth, height, width = shape
    patch_depth, patch_height, patch_width = patch_size

    margin_depth = patch_depth // 2
    margin_height = patch_height // 2
    margin_width = patch_width // 2

    z_coords = np.arange(margin_depth, depth - margin_depth)
    y_coords = np.arange(margin_height, height - margin_height)
    x_coords = np.arange(margin_width, width - margin_width)

    total = len(z_coords) * len(y_coords) * len(x_coords)
    stride = int((total / num_samples) ** (1 / 3))
    stride = max(stride, 1)

    z_coords = z_coords[::stride]
    y_coords = y_coords[::stride]
    x_coords = x_coords[::stride]

    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    centers = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)

    if len(centers) > num_samples:
        centers = centers[:num_samples]

    return centers


def process_case(task):
    case_name, idx, image_root, label_root, save_image_root, save_label_root, patch_size, patch_per_case, uniform_ratio, seed = task
    rng = np.random.default_rng(seed + idx)

    image_path = os.path.join(image_root, case_name)
    label_path = os.path.join(label_root, case_name.replace("images", "labels"))

    image = load_npy(image_path)
    label = load_npy(label_path)

    case_id = case_name.replace(".npy", "")

    patch_depth, patch_height, patch_width = patch_size

    num_uniform = int(patch_per_case * uniform_ratio)
    num_random = patch_per_case - num_uniform

    valid_centers = get_valid_centers(label.shape, patch_size)
    centers_uniform = get_uniform_centers(label.shape, patch_size, num_uniform)
    centers_random = valid_centers[
        rng.choice(len(valid_centers), num_random, replace=False)
    ]
    centers = np.concatenate([centers_uniform, centers_random], axis=0)

    for patch_idx, center in enumerate(centers):
        z_start, y_start, x_start = center - np.array(patch_size) // 2

        image_patch = image[
            z_start:z_start + patch_depth,
            y_start:y_start + patch_height,
            x_start:x_start + patch_width,
        ]
        label_patch = label[
            z_start:z_start + patch_depth,
            y_start:y_start + patch_height,
            x_start:x_start + patch_width,
        ]

        patch_name = f"{case_id}__patch_{patch_idx:03d}.npy"

        np.save(os.path.join(save_image_root, patch_name), image_patch.astype(np.float32))
        np.save(os.path.join(save_label_root, patch_name), label_patch.astype(np.uint8))

    return 1


def main():
    args = parse_args()
    preprocessed_dir = Path(args.preprocessed_dir)
    image_root = str(preprocessed_dir / "images")
    label_root = str(preprocessed_dir / "labels")
    save_image_root = str(preprocessed_dir / "patches_images")
    save_label_root = str(preprocessed_dir / "patches_labels")
    patch_size = tuple(args.patch_size)
    patch_per_case = args.patch_per_case
    uniform_ratio = args.uniform_ratio
    num_workers = args.num_workers

    os.makedirs(save_image_root, exist_ok=True)
    os.makedirs(save_label_root, exist_ok=True)

    case_list = [file_name for file_name in os.listdir(image_root) if file_name.endswith(".npy")]
    case_list = sorted(case_list)

    tasks = [
        (
            case_name,
            idx,
            image_root,
            label_root,
            save_image_root,
            save_label_root,
            patch_size,
            patch_per_case,
            uniform_ratio,
            args.seed,
        )
        for idx, case_name in enumerate(case_list)
    ]

    print(f"Total cases: {len(case_list)}")

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_case, tasks), total=len(tasks)))

    print("Patch generation done.")


if __name__ == "__main__":
    main()

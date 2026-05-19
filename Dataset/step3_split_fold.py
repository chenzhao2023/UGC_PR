import argparse
import os
import random
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = DATASET_DIR / "Preprocessed"


def parse_args():
    parser = argparse.ArgumentParser(description="Create deterministic K-fold splits for 3D NPY volumes.")
    parser.add_argument("--preprocessed_dir", type=str, default=str(DEFAULT_PREPROCESSED_DIR))
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_k(items, num_folds):
    fold_size = len(items) // num_folds
    folds = []

    for fold_idx in range(num_folds):
        if fold_idx < num_folds - 1:
            folds.append(items[fold_idx * fold_size:(fold_idx + 1) * fold_size])
        else:
            folds.append(items[fold_idx * fold_size:])

    return folds


def main():
    args = parse_args()
    preprocessed_dir = Path(args.preprocessed_dir)
    image_dir = preprocessed_dir / "images"
    split_dir = preprocessed_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".npy") and file_name.startswith("images"):
            all_images.append(file_name)

    print("Total images:", len(all_images))

    random.seed(args.seed)
    random.shuffle(all_images)

    folds = split_k(all_images, args.num_folds)

    for fold_idx in range(args.num_folds):
        val_images = folds[fold_idx]
        train_images = []

        for other_fold_idx in range(args.num_folds):
            if other_fold_idx != fold_idx:
                train_images += folds[other_fold_idx]

        with open(split_dir / f"fold{fold_idx + 1}_train.txt", "w", encoding="utf-8") as file_obj:
            for image_name in train_images:
                file_obj.write(image_name + "\n")

        with open(split_dir / f"fold{fold_idx + 1}_val.txt", "w", encoding="utf-8") as file_obj:
            for image_name in val_images:
                file_obj.write(image_name + "\n")

        print(f"Fold {fold_idx + 1}: train={len(train_images)}, val={len(val_images)}")

    print(f"\n{args.num_folds}-fold split done: {split_dir}")


if __name__ == "__main__":
    main()

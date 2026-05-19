import os
import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npy(path):
    return np.load(path, mmap_mode="r")


def load_case_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


class OfflineMRAPatchDataset(Dataset):
    def __init__(
        self,
        root_images,
        root_labels,
        case_list,
        samples_per_case=32,
        base_seed=42,
        shuffle=True,
    ):
        self.root_images = root_images
        self.root_labels = root_labels
        self.samples_per_case = samples_per_case
        self.base_seed = base_seed
        self.shuffle = shuffle

        patch_files = [file_name for file_name in os.listdir(root_images) if file_name.endswith(".npy")]
        case_ids = [case_id.replace(".npy", "") for case_id in case_list]
        case_set = set(case_ids)

        self.files = [file_name for file_name in patch_files if file_name.split("__")[0] in case_set]

        case_to_indices = defaultdict(list)
        for index, file_name in enumerate(self.files):
            case_to_indices[file_name.split("__")[0]].append(index)

        self.case_to_indices = dict(case_to_indices)
        self.case_ids = list(self.case_to_indices.keys())

        print(f"[Dataset] cases={len(self.case_ids)}, patches={len(self.files)}")

        self.epoch = 0
        self.selected_indices = []
        self._resample()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._resample()

    def _resample(self):
        rng = random.Random(self.base_seed + self.epoch)
        selected_indices = []

        for case_id in self.case_ids:
            case_indices = self.case_to_indices[case_id]
            if len(case_indices) >= self.samples_per_case:
                sampled_indices = rng.sample(case_indices, self.samples_per_case)
            else:
                sampled_indices = rng.choices(case_indices, k=self.samples_per_case)
            selected_indices.extend(sampled_indices)

        if self.shuffle:
            rng.shuffle(selected_indices)

        self.selected_indices = selected_indices

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        file_name = self.files[self.selected_indices[index]]
        image = np.load(os.path.join(self.root_images, file_name))
        label = np.load(os.path.join(self.root_labels, file_name))

        return {
            "case_id": file_name.split("__")[0],
            "img": torch.from_numpy(image).unsqueeze(0),
            "label": torch.from_numpy(label).long(),
        }


class MRAPatchDataset(Dataset):
    def __init__(
        self,
        case_list,
        root_images,
        root_labels,
        patch_size=(64, 64, 64),
        samples_per_case=16,
        cache_size=24,
    ):
        self.case_list = case_list
        self.root_images = root_images
        self.root_labels = root_labels
        self.patch_size = np.array(patch_size)
        self.samples_per_case = samples_per_case
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.case_list) * self.samples_per_case

    def _load_case(self, case_name):
        if case_name in self.cache:
            self.cache.move_to_end(case_name)
            return self.cache[case_name]

        image = load_npy(os.path.join(self.root_images, case_name))
        label = load_npy(os.path.join(self.root_labels, case_name.replace("images", "labels")))
        valid_centers = self._compute_valid_centers(label)

        if valid_centers.shape[0] == 0:
            raise RuntimeError(f"No valid centers in {case_name}")

        case_data = {
            "img": image,
            "label": label,
            "valid_centers": valid_centers,
        }

        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)

        self.cache[case_name] = case_data
        return case_data

    def _compute_valid_centers(self, label):
        depth, height, width = label.shape
        patch_depth, patch_height, patch_width = self.patch_size

        z_coords = np.arange(patch_depth // 2, depth - patch_depth // 2)
        y_coords = np.arange(patch_height // 2, height - patch_height // 2)
        x_coords = np.arange(patch_width // 2, width - patch_width // 2)

        zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        return np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)

    def __getitem__(self, index):
        case_index = index // self.samples_per_case
        case_name = self.case_list[case_index]

        case_data = self._load_case(case_name)
        center = case_data["valid_centers"][random.randrange(len(case_data["valid_centers"]))]

        patch_depth, patch_height, patch_width = self.patch_size
        z_start, y_start, x_start = center - self.patch_size // 2

        image_patch = case_data["img"][
            z_start:z_start + patch_depth,
            y_start:y_start + patch_height,
            x_start:x_start + patch_width,
        ]
        label_patch = case_data["label"][
            z_start:z_start + patch_depth,
            y_start:y_start + patch_height,
            x_start:x_start + patch_width,
        ]

        return {
            "case_id": case_name,
            "img": torch.from_numpy(image_patch.copy()).unsqueeze(0),
            "label": torch.from_numpy(label_patch.copy()).long(),
        }


class MRACaseDataset(Dataset):
    def __init__(self, case_list, root_images, root_labels):
        self.case_list = case_list
        self.root_images = root_images
        self.root_labels = root_labels

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        case_name = self.case_list[index]
        image = load_npy(os.path.join(self.root_images, case_name)).copy()
        label = load_npy(os.path.join(self.root_labels, case_name.replace("images", "labels"))).copy()

        return {
            "case_id": case_name,
            "img": torch.from_numpy(image).unsqueeze(0),
            "label": torch.from_numpy(label).long(),
        }

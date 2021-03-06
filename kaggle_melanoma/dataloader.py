from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import albumentations as albu
import numpy as np
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):
    def __init__(self, samples: List[Tuple[Union[Path, str], int, str]], transform: albu.Compose) -> None:
        """

        Args:
            samples: List of pairs [(path_to_image_file, target), ...]
            transform:
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, target, group = self.samples[idx]

        image = load_rgb(image_path, lib="jpeg4py")

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {
            "image_id": Path(image_path).stem,
            "features": tensor_from_rgb_image(image),
            "targets": torch.from_numpy(np.array([target])),
            "group": group,
        }


class MelanomaTestDataset(Dataset):
    def __init__(self, samples: List[Tuple[Union[Path, str], str]], transform: albu.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, group = self.samples[idx]

        image = load_rgb(image_path, lib="jpeg4py")

        # apply augmentations
        image = self.transform(image=image)["image"]

        return {"image_id": Path(image_path).stem, "features": tensor_from_rgb_image(image), "group": group}

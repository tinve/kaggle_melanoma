from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional, Any
from sklearn import metrics
import numpy as np
import pandas as pd
import re
import torch


def get_samples(data_path: Path) -> List[Tuple[Path, int]]:
    df = pd.read_csv(data_path / "train.csv")[["image_name", "target"]]
    df["image_name"] = df["image_name"].apply(lambda x: data_path / "jpeg" / "train" / (x + ".jpg"))
    return list(df.itertuples(index=False, name=None))


def melanoma_auc(y_true: Union[np.ndarray, list], y_valid: Union[np.ndarray, list]) -> float:
    y_true = np.array(y_true)
    y_valid = np.array(y_valid)
    return metrics.roc_auc_score(y_true, y_valid)


def cross_entropy(predictions, targets):
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(targets * np.log(predictions))


def load_checkpoint(file_path: Union[Path, str], rename_in_layers: Optional[dict] = None) -> Dict[str, Any]:
    """Loads PyTorch checkpoint, optionally renaming layer names.

    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if rename_in_layers is not None:
        model_state_dict = checkpoint["state_dict"]

        result = {}
        for key, value in model_state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = re.sub(key_r, value_r, key)

            result[key] = value

        checkpoint["state_dict"] = result

    return checkpoint

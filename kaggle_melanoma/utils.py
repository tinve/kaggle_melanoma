from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional, Any, Generator, DefaultDict, Counter, Set

# import typing
from sklearn import metrics
import re
import torch

import random
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter as collections_Counter


def stratified_group_k_fold(target: np.ndarray, groups: np.ndarray, num_folds: int, seed: Optional[int]) -> Generator:
    labels_num = np.max(target) + 1
    y_counts_per_group: DefaultDict[str, np.ndarray] = defaultdict(lambda: np.zeros(labels_num))
    y_distribution: Counter[int] = collections_Counter()
    for t, g in zip(target, groups):
        y_counts_per_group[g][t] += 1
        y_distribution[t] += 1

    y_counts_per_fold: DefaultDict[int, np.ndarray] = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold: DefaultDict[int, Set[str]] = defaultdict(set)

    def eval_y_counts_per_fold(y_counts: int, fold: int) -> float:
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distribution[label] for i in range(num_folds)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = -1
        min_eval = -1.0
        for i in range(num_folds):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval < 0 or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(num_folds):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_samples(data_path: Path) -> List[Tuple[Path, int, str]]:
    df = pd.read_csv(data_path / "train.csv")[["image_name", "target", "patient_id"]]
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

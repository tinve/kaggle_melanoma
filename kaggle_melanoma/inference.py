import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.core.serialization import from_dict
from pytorch_toolbelt.inference import tta
from sklearn.metrics import log_loss
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kaggle_melanoma.dataloader import MelanomaDataset, MelanomaTestDataset
from kaggle_melanoma.train import Melanoma
from kaggle_melanoma.utils import get_samples, load_checkpoint, melanoma_auc, stratified_group_k_fold


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-w", "--checkpoint_path", type=Path, help="Path to weights.", required=True)
    arg("-o", "--output_path", type=Path, help="Output_path.", required=True)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    model = Melanoma(hparams=hparams)

    corrections: Dict[str, str] = {}

    checkpoint = load_checkpoint(file_path=args.checkpoint_path, rename_in_layers=corrections)  # type: ignore

    model.load_state_dict(checkpoint["state_dict"])

    model = nn.Sequential(model, nn.Sigmoid())

    model.eval()
    model = model.half()
    model.cuda()

    with torch.no_grad():
        print("Evaluate on validation.")
        val_aug = from_dict(hparams["val_aug"])
        val_samples = np.array(get_samples(Path(hparams["data_path"])))
        val_target = val_samples[:, 1]
        val_groups = val_samples[:, 2]

        kf = stratified_group_k_fold(val_target, val_groups, num_folds=hparams["num_folds"], seed=hparams["seed"])

        for fold_id, (_, val_index) in enumerate(kf):
            if fold_id != hparams["fold_id"]:
                continue

            val_samples = val_samples[val_index].tolist()

        dataloader = DataLoader(
            MelanomaDataset(val_samples, val_aug),
            batch_size=hparams["val_parameters"]["batch_size"],
            num_workers=hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        y_true = []
        y_pred = []

        for batch in tqdm(dataloader):
            features = batch["features"]
            targets = batch["targets"]

            preds = model(features.half().cuda())

            y_pred += preds.cpu().numpy().T.tolist()[0]
            y_true += targets.cpu().numpy().T.tolist()[0]

        print("Val log loss = ", log_loss(y_true, y_pred))
        print("Auc = ", melanoma_auc(y_true, y_pred))

        print("Evaluate on test.")
        test_aug = from_dict(hparams["test_aug"])

        test_file_names = sorted((Path(hparams["data_path"]) / "jpeg" / "test").glob("*.jpg"))

        dataloader = DataLoader(
            MelanomaTestDataset(test_file_names, test_aug),
            batch_size=hparams["test_parameters"]["batch_size"],
            num_workers=hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        y_pred = []
        file_ids = []

        for batch in tqdm(dataloader):
            file_ids += batch["image_id"]
            features = batch["features"]

            if hparams["test_parameters"]["tta"] == "d4":
                preds = tta.d4_image2label(model, features.half().cuda())
            elif hparams["test_parameters"]["tta"] == "lr":
                preds = tta.fliplr_image2label(model, features.half().cuda())
            else:
                preds = model(features.half().cuda())

            y_pred += preds.cpu().numpy().T.tolist()[0]

        submission = pd.DataFrame({"image_name": file_ids, "target": y_pred})

        submission.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()

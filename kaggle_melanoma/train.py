import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import apex
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from pytorch_lightning.logging import NeptuneLogger
from torch.utils.data import DataLoader

from kaggle_melanoma.dataloader import MelanomaDataset
from kaggle_melanoma.utils import get_samples, melanoma_auc, stratified_group_k_fold


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-r", "--resume", type=Path, help="Path to the checkpoint.")
    return parser.parse_args()


class Melanoma(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.model = object_from_dict(self.hparams["model"])
        if hparams["sync_bn"]:
            self.model = apex.parallel.convert_syncbn_model(self.model)

        self.loss = object_from_dict(self.hparams["loss"])
        self.train_samples = []  # skipcq: PYL-W0201
        self.val_samples = []  # skipcq: PYL-W0201

    def forward(self, batch: Dict) -> torch.Tensor:  # skipcq: PYL-W0221
        return self.model(batch)

    def prepare_data(self):
        samples = np.array(get_samples(Path(self.hparams["data_path"])))
        target = samples[:, 1]
        groups = samples[:, 2]

        kf = stratified_group_k_fold(target, groups, num_folds=self.hparams["num_folds"], seed=self.hparams["seed"])

        for fold_id, (train_index, val_index) in enumerate(kf):
            if fold_id != self.hparams["fold_id"]:
                continue

            self.train_samples = samples[train_index].tolist()
            self.val_samples = samples[val_index].tolist()

    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])

        result = DataLoader(
            MelanomaDataset(self.train_samples, train_aug),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])

        return DataLoader(
            MelanomaDataset(self.val_samples, val_aug),
            batch_size=self.hparams["val_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"], params=filter(lambda x: x.requires_grad, self.model.parameters()),
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]  # skipcq: PYL-W0201

        return self.optimizers, [scheduler]

    # skipcq: PYL-W0613, PYL-W0221
    def training_step(self, batch, batch_idx):
        features = batch["features"]
        logits = self.forward(features)

        total_loss = self.loss(logits, batch["targets"])

        logs = {"train_loss": total_loss, "lr": self._get_current_lr()}

        return {"loss": total_loss, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()

    # skipcq: PYL-W0613, PYL-W0221
    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        targets = batch["targets"]

        logits = self.forward(features)

        return {"val_loss": self.loss(logits, targets), "logits": logits, "targets": targets}

    def validation_epoch_end(self, outputs: List) -> Dict[str, Any]:
        result_logits: List[float] = []
        result_targets: List[float] = []

        for output in outputs:
            result_logits += torch.sigmoid(output["logits"]).cpu().numpy().flatten().tolist()
            result_targets += output["targets"].cpu().numpy().flatten().tolist()

        auc_score = melanoma_auc(result_targets, result_logits)

        loss = find_average(outputs, "val_loss")
        logs = {"val_loss": loss, "epoch": self.trainer.current_epoch, "auc_score": auc_score}

        return {"val_loss": loss, "log": logs}


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    # csv_logger = CsvLogger(
    #     train_csv_path=Path(hparams["experiment_name"]) / "train.csv",
    #     val_csv_path=Path(hparams["experiment_name"]) / "val.csv",
    #     train_columns=["train_loss", "lr"],
    #     val_columns=["val_loss", "auc_score"],
    # )

    # neptune_logger = NeptuneLogger(
    #     api_key=os.environ["NEPTUNE_API_TOKEN"],
    #     project_name="tinve/kaggle-melanoma",
    #     experiment_name=f"{hparams['experiment_name']}",  # Optional,
    #     tags=["pytorch-lightning", "mlp"],  # Optional,
    #     upload_source_files=[],
    # )

    pipeline = Melanoma(hparams)

    Path(hparams["checkpoint_callback"]["filepath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"],
        # logger=neptune_logger,
        checkpoint_callback=object_from_dict(hparams["checkpoint_callback"]),
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()

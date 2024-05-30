import os
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm import tqdm

from unet.train_dataset import DeadwoodDataset
from unet.unet_loss import BCEDiceLoss
from unet.unet_model import UNet


class DeadwoodTrainer:

    default_config = {
        "use_wandb": True,
        "epochs": 60,
        "no_folds": 3,
        "batch_size": 64,
        "epoch_train_samples": 50000,
        "epoch_val_samples": 100000,
        "test_size": 0,
        "balancing_factor": 1,
        "pos_weight": 40.0,
        "bce_weight": 0.2,
        "bins": np.arange(0, 0.21, 0.02),
        "amp": True,
        "learning_rate": 1e-5,
        "weight_decay": 1e-8,
        "momentum": 0.999,
        "lr_patience": 5,
        "num_workers": 32,
        "gradient_clipping": 1.0,
        "experiments_dir": "/net/home/jmoehring/experiments",
        "register_file": "/net/scratch/jmoehring/tiles_register_biome_bin.csv",
        "random_seed": 100,
    }

    def __init__(self, run_name: str, config=default_config):
        self.config = config
        self.run_name = run_name

    def setup_device(self):
        # preferably use GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_model(self):
        # model with three input channels (RGB)
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        if torch.cuda.device_count() > 1:
            # train on GPU 1 and 2
            model = nn.DataParallel(model)
        model.to(device=self.device)
        self.model = model

        if self.config["use_wandb"]:
            wandb.watch(model, log="all")

        self.criterion = BCEDiceLoss()

        # loss function (binary cross entropy)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([self.config["pos_weight"]]).to(device=self.device)
        )

        # optimizer
        self.optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            momentum=self.config["momentum"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=self.config["lr_patience"]
        )
        self.grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(
            enabled=self.config["amp"]
        )

    def setup_dataset(self):
        register_df = pd.read_csv(self.config["register_file"])
        self.dataset = DeadwoodDataset(
            register_df=register_df,
            no_folds=self.config["no_folds"],
            random_seed=self.config["random_seed"],
            bins=self.config["bins"],
            test_size=self.config["test_size"],
        )

    def setup_dataloader(self, fold: int):
        train_set, val_set = self.dataset.get_train_val_fold(fold)
        train_sampler = WeightedRandomSampler(
            self.dataset.get_train_sample_weights(
                fold=fold, balancing_factor=self.config["balancing_factor"]
            ).tolist(),  # Convert tensor to sequence of floats
            self.config["epoch_train_samples"],
            replacement=False,
        )
        loader_args = {
            "batch_size": self.config["batch_size"],
            "num_workers": self.config["num_workers"],
            "pin_memory": True,
            "shuffle": False,
        }
        self.train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
        self.val_loader = DataLoader(val_set, **loader_args)

        if self.config["epoch_val_samples"] > 0:
            val_sampler = RandomSampler(
                val_set, replacement=False, num_samples=self.config["epoch_val_samples"]
            )
            self.val_loader = DataLoader(val_set, sampler=val_sampler, **loader_args)

    def setup_experiment(self, run_name: str):
        self.run_dir = Path(self.config["experiments_dir"]) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.config["use_wandb"]:
            self.experiment = wandb.init(
                project="standing-deadwood-unet-pro",
                resume="allow",
                name=run_name,
                config=self.config,
            )

    def train(self, fold: int):
        for epoch in range(self.config["epochs"]):
            self.model.train()
            epoch_loss = 0
            pbar = tqdm(
                total=len(self.train_loader) * self.config["batch_size"],
                desc=f"Training: Epoch {epoch}/{self.config['epochs']}",
                unit="img",
            )
            for images, masks_true, _ in self.train_loader:
                images = images.to(
                    device=self.device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                masks_true = masks_true.to(
                    device=self.device, dtype=torch.long
                ).squeeze()

                with torch.amp.autocast(
                    self.device.type if self.device.type != "mps" else "cpu",
                    enabled=self.config["amp"],
                ):
                    masks_pred = self.model(images).squeeze(1)
                    loss = self.criterion(masks_pred.squeeze(1), masks_true.float())
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clipping"]
                    )
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                    epoch_loss += loss.item()
                    step += 1
                    pbar.update(images.shape[0])

            train_loss = epoch_loss / len(self.train_loader)
            val_loss = self.evaluate(epoch=epoch)
            self.scheduler.step(val_loss)

            if self.config["use_wandb"]:
                self.experiment.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch": epoch,
                        "fold": fold,
                    }
                )

    @torch.inference_mode()
    def evaluate(self, epoch: int):
        self.model.eval()
        epoch_loss = 0
        with torch.autocast(
            self.device.type if self.device.type != "mps" else "cpu",
            enabled=self.config["amp"],
        ):
            pbar = tqdm(
                total=len(self.val_loader) * self.config["batch_size"],
                desc=f"Validation: Epoch {epoch}/{self.config['epochs']}",
                unit="img",
            )
            for images, masks_true, _ in self.val_loader:
                images = images.to(
                    device=self.device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                masks_true = masks_true.to(
                    device=self.device, dtype=torch.long
                ).squeeze()

                masks_pred = self.model(images).squeeze()
                loss = self.criterion(masks_pred, masks_true.float())
                epoch_loss += loss.item()
                pbar.update(images.shape[0])

        return epoch_loss / len(self.val_loader)

    def save_checkpoint(self, fold: int, epoch: int):
        checkpoint_path = (
            Path(self.config["experiments_dir"]) / f"fold_{fold}_epoch_{epoch}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "grad_scaler_state_dict": self.grad_scaler.state_dict(),
            },
            checkpoint_path,
        )

    def run(self):
        self.setup_experiment(run_name=self.run_name)
        self.setup_dataset()

        for fold in range(self.config["no_folds"]):

            self.setup_device()
            self.setup_model()
            self.setup_dataloader(fold=fold)
            self.train(fold=fold)

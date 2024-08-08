from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
from accelerate import Accelerator

from unet.train_dataset import DeadwoodDataset
from unet.unet_loss import BCEDiceLoss, GroupedConfusion
import segmentation_models_pytorch as smp


class DeadwoodTrainer:

    def __init__(self, run_name: str, config):
        self.config = config
        self.run_name = run_name
        self.accelerator = Accelerator(
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ],
        )

        if self.config["run_fold"] >= 0:
            self.run_name = f"{run_name}_fold{self.config['run_fold']}"
            self.range_folds = [self.config["run_fold"]]
        else:
            self.range_folds = range(self.config["no_folds"])

    def setup_device(self):
        self.device = self.accelerator.device

    def setup_model(self):
        model = smp.Unet(
            encoder_name=self.config["encoder_name"],
            encoder_weights=self.config["encoder_weights"],
            in_channels=3,
            classes=1,
        )
        self.model = model.to(self.device)
        self.criterion = BCEDiceLoss(
            pos_weight=torch.Tensor([self.config["pos_weight"]]).to(
                self.device, torch.float32
            ),
            bce_weight=self.config["bce_weight"],
        )
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            momentum=self.config["momentum"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=self.config["lr_patience"]
        )

    def setup_dataset(self):
        register_df = pd.read_csv(self.config["register_file"])
        self.dataset = DeadwoodDataset(
            register_df=register_df,
            images_dir=self.config["images_dir"],
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
            ).tolist(),
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

    def setup_accelerator(self):
        model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

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
            self.val_table = wandb.Table(
                columns=[
                    "fold",
                    "epoch",
                    "biome",
                    "resolution_bin",
                    "precision",
                    "recall",
                    "f1",
                    "positives",
                    "negatives",
                ]
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
            for images, masks_true, masks_weights, _ in self.train_loader:
                self.optimizer.zero_grad()

                masks_pred = self.model(images)
                loss = self.criterion(
                    masks_pred.squeeze(1),
                    masks_true.float(),
                    masks_weights.squeeze(1),
                )

                self.accelerator.backward(loss)
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.update(images.shape[0])

            train_loss = epoch_loss / len(self.train_loader)
            val_loss, metrics_df = self.evaluate(epoch=epoch, fold=fold)
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
                for _, row in metrics_df.iterrows():
                    self.val_table.add_data(
                        row["fold"],
                        row["epoch"],
                        row["biome"],
                        row["resolution_bin"],
                        row["precision"],
                        row["recall"],
                        row["f1"],
                        row["positives"],
                        row["negatives"],
                    )

            if self.config["save_checkpoint"]:
                self.save_checkpoint(fold=fold, epoch=epoch)

    @torch.inference_mode()
    def evaluate(self, fold: int, epoch: int):
        self.model.eval()
        epoch_loss = 0
        confusion = GroupedConfusion()

        pbar = tqdm(
            total=len(self.val_loader) * self.config["batch_size"],
            desc=f"Validation: Epoch {epoch}/{self.config['epochs']}",
            unit="img",
        )
        for images, masks_true, masks_weights, metas in self.val_loader:
            masks_pred = self.model(images).squeeze()
            
            all_gathered = self.accelerator.gather_for_metrics(
                (masks_pred, masks_true, masks_weights, metas),
            )
            # Print the length and elements to debug
            print(f"Number of elements returned: {len(all_gathered)}")
            # print(f"Elements returned: {all_gathered}")

            # loss = self.criterion(
            #     all_masks_pred,
            #     all_masks_true.squeeze().float(),
            #     all_masks_weights.squeeze(),
            # )
            # epoch_loss += loss.item()

            # confusion(all_masks_pred, all_masks_true, all_metas)
            pbar.update(images.shape[0])

        metrics_df = confusion.compute_metrics(fold, epoch)
        return epoch_loss / len(self.val_loader), metrics_df

    def save_checkpoint(self, fold: int, epoch: int):
        checkpoint_path = (
            Path(self.config["experiments_dir"])
            / self.run_name
            / f"fold_{fold}_epoch_{epoch}"
        )
        self.accelerator.wait_for_everyone()
        self.accelerator.save(self.model.state_dict(), checkpoint_path)

    def run(self):
        self.setup_experiment(run_name=self.run_name)
        self.setup_dataset()

        for fold in self.range_folds:
            self.setup_device()
            self.setup_model()
            self.setup_dataloader(fold=fold)
            self.setup_accelerator()
            self.train(fold=fold)

        if self.config["use_wandb"]:
            self.experiment.log({"val_metrics": self.val_table})

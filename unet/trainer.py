from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from tqdm import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs
from unet.train_dataset import DeadwoodDataset
from unet.unet_loss import BCEDiceLoss,PrecisionRecallF1
import segmentation_models_pytorch as smp


class DeadwoodTrainer:

    def __init__(self, run_name: str, config):
        self.config = config
        self.run_name = run_name
        self.run_path = Path(self.config["experiments_dir"]) / self.run_name
        
        if self.config["run_fold"] >= 0:
            self.run_name = f"{run_name}_fold{self.config['run_fold']}"
            self.range_folds = [self.config["run_fold"]]
        else:
            self.range_folds = range(self.config["no_folds"])
        

    def setup_device(self):
        self.accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)], log_with="wandb", project_dir=self.config["experiments_dir"])
        self.device = self.accelerator.device

    def setup_model(self):
        # model with three input channels (RGB)
        model = smp.Unet(
            encoder_name=self.config["encoder_name"],
            encoder_weights=self.config["encoder_weights"],
            in_channels=3,
            classes=1,
        )
        model.to(device=self.device)
        self.model = model
        self.criterion = BCEDiceLoss(
            pos_weight=torch.Tensor([self.config["pos_weight"]]).to(
                self.device, torch.float32
            ),
            bce_weight=self.config["bce_weight"],
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
            self.accelerator.init_trackers("standing-deadwood-unet-pro", config=self.config, init_kwargs={"wandb":{"name":self.run_name}})
            self.accelerator.get_tracker("wandb").define_metric("train_loss", summary="min")
            self.accelerator.get_tracker("wandb").define_metric("val_loss", summary="min")

    def setup_accelerator(self):
        model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler

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
                images = images.to(
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                masks_true = masks_true.to(dtype=torch.long).squeeze()
                masks_weights = masks_weights.to(dtype=torch.uint8).squeeze()

                with torch.amp.autocast(
                    self.device.type if self.device.type != "mps" else "cpu",
                    enabled=self.config["amp"],
                ):
                    masks_pred = self.model(images).squeeze(1)
                    loss = self.criterion(
                        masks_pred.squeeze(1),
                        masks_true.float(),
                        masks_weights.squeeze(1),
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.accelerator.backward(loss)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clipping"]
                    )

                    epoch_loss += loss.item()
                    pbar.update(images.shape[0])

            train_loss = epoch_loss / len(self.train_loader)
            val_loss, metrics = self.evaluate(epoch=epoch, fold=fold)
            self.scheduler.step(val_loss)

            if self.config["use_wandb"]:
                self.accelerator.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "metrics": metrics,
                        "epoch": epoch,
                        "fold": fold,
                    }
                )

            if self.config["save_checkpoint"]:
                self.save_checkpoint(fold=fold, epoch=epoch)

    @torch.inference_mode()
    def evaluate(self, fold: int, epoch: int):
        self.model.eval()
        epoch_loss = 0

        pbar = tqdm(
            total=len(self.val_loader) * self.config["batch_size"],
            desc=f"Validation: Epoch {epoch}/{self.config['epochs']}",
            unit="img",
        )
        metrics_eval = []
        for images, masks_true, masks_weights, indexes in self.val_loader:
            images = images.to(
                dtype=torch.float32,
                memory_format=torch.channels_last,
            )
            masks_true = masks_true.to(dtype=torch.long).squeeze()
            masks_weights = masks_weights.to(dtype=torch.uint8).squeeze()
            masks_pred = self.model(images).squeeze()
            
            all_masks_true = self.accelerator.gather(masks_true)
            all_masks_pred = self.accelerator.gather(masks_pred)
            all_masks_weights = self.accelerator.gather(masks_weights)
            all_indexes = self.accelerator.gather(indexes)
            
            loss = self.criterion(
                all_masks_pred, all_masks_true.float(), all_masks_weights.squeeze(1)
            )
            epoch_loss += loss.item()
            all_precision, all_recall, all_f1 = PrecisionRecallF1()(all_masks_true, all_masks_pred, all_masks_weights)
            all_metrics = torch.cat((all_precision, all_recall, all_f1, all_indexes.unsqueeze(1).cpu()), dim=1)
            metrics_eval.append(all_metrics)
            pbar.update(images.shape[0])

        metrics = torch.cat(metrics_eval, dim=0)
        print(metrics)
        print(metrics.shape)
        return epoch_loss / len(self.val_loader), metrics

    def save_checkpoint(self, fold: int, epoch: int):
        checkpoint_name = f"{self.run_path}/fold_{fold}_epoch_{epoch}"
        self.accelerator.save_state(output_dir=checkpoint_name)
        

    def run(self):
        self.setup_device()
        self.setup_experiment(run_name=self.run_name)
        self.setup_dataset()

        for fold in self.range_folds:
            self.setup_model()
            self.setup_dataloader(fold=fold)
            self.setup_accelerator()
            self.train(fold=fold)
        
        self.accelerator.end_training()
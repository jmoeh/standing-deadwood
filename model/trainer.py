from datetime import timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from accelerate import Accelerator
from model.cappedsampler import CappedSampler
from model.train_dataset import DeadwoodDataset
from model.loss import GlobalPrecisionRecallF1IoU, TverskyFocalLoss, BCELoss, DiceLoss
import segmentation_models_pytorch as smp
from accelerate.utils import InitProcessGroupKwargs


class DeadwoodTrainer:

    def __init__(self, run_name: str, config):
        self.config = config
        self.run_name = run_name
        self.run_path = Path(self.config["experiments_dir"]) / self.run_name
        self.run_path.mkdir(parents=True, exist_ok=True)

        if self.config["run_fold"] >= 0:
            self.run_name = f"{run_name}_fold{self.config['run_fold']}"
            self.range_folds = [self.config["run_fold"]]
        else:
            self.range_folds = range(self.config["no_folds"])

        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        torch.manual_seed(self.config["random_seed"])
        torch.cuda.manual_seed_all(self.config["random_seed"])

    def setup_device(self):
        self.accelerator = Accelerator(
            log_with="wandb",
            project_dir=self.config["experiments_dir"],
            gradient_accumulation_steps=self.config["gradient_accumulation"],
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(seconds=800000))
            ],
        )
        self.device = self.accelerator.device

    def lr_lambda(self, epoch):
        if epoch < self.config["warmup_epochs"]:
            # Return a multiplier that gradually increases from warmup_start_lr/learning_rate to 1
            return (
                (self.config["learning_rate"] - self.config["warmup_start_lr"])
                * (epoch / self.config["warmup_epochs"]) +
                self.config["warmup_start_lr"]) / self.config["learning_rate"]
        return 1.0  # After warmup, maintain the base learning rate

    def setup_model(self):
        # model with three input channels (RGB)
        model = smp.Unet(
            encoder_name=self.config["encoder_name"],
            encoder_weights=self.config["encoder_weights"],
            in_channels=3,
            classes=1,
        ).to(memory_format=torch.channels_last)
        self.model = torch.compile(model)

        if self.config["loss"] == "bce":
            self.criterion = BCELoss(pos_weight=torch.Tensor(
                [self.config["pos_weight"]]).to(self.device, torch.float32))
        elif self.config["loss"] == "tverskyfocal":
            self.criterion = TverskyFocalLoss(
                alpha=self.config["alpha"],
                beta=self.config["beta"],
                gamma=self.config["gamma"],
            )
        elif self.config["loss"] == "dice":
            self.criterion = TverskyFocalLoss(alpha=0.5, beta=0.5, gamma=1)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_lambda)

    def setup_dataset(self):
        register_df = pd.read_csv(self.config["register_file"])
        self.dataset = DeadwoodDataset(
            register_df=register_df,
            images_dir=self.config["images_dir"],
            no_folds=self.config["no_folds"],
            random_seed=self.config["random_seed"],
            bins=self.config["bins"],
            test_size=self.config["test_size"],
            verbose=True,
        )

    def setup_dataloader(self, fold: int):
        _, val_set = self.dataset.get_train_val_fold(fold)
        train_sampler = CappedSampler(dataset=self.dataset, fold=fold)

        loader_args = {
            "batch_size": self.config["batch_size"],
            "num_workers": self.config["num_workers"],
            "pin_memory": True,
            "shuffle": False,
        }
        self.train_loader = DataLoader(self.dataset,
                                       sampler=train_sampler,
                                       **loader_args)
        self.val_loader = DataLoader(val_set, **loader_args)

        if self.config["epoch_val_samples"] > 0:
            val_sampler = RandomSampler(
                val_set,
                replacement=True,
                num_samples=self.config["epoch_val_samples"])
            self.val_loader = DataLoader(val_set,
                                         sampler=val_sampler,
                                         **loader_args)

    def setup_experiment(self, run_name: str):
        self.metrics_df = pd.DataFrame()
        if self.config["use_wandb"]:
            self.accelerator.init_trackers(
                "standing-deadwood-unet-pro",
                config=self.config,
                init_kwargs={"wandb": {
                    "name": self.run_name
                }},
            )

    def setup_accelerator(self):
        model, optimizer, train_loader, val_loader, scheduler = (
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_loader,
                self.val_loader,
                self.scheduler,
            ))
        self.model = model
        self.model.to(self.device)
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
                disable=not self.accelerator.is_local_main_process,
            )
            for images, masks_true, masks_weights, _ in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad(set_to_none=True)
                    images = images.to(
                        dtype=torch.float32,
                        memory_format=torch.channels_last,
                    )
                    masks_true = masks_true.to(dtype=torch.long).squeeze()
                    masks_weights = masks_weights.to(
                        dtype=torch.uint8).squeeze()

                    masks_pred = self.model(images).squeeze(1)
                    loss = self.criterion(
                        masks_pred.squeeze(1),
                        masks_true.float(),
                        masks_weights.squeeze(1),
                    )
                    self.accelerator.backward(loss.contiguous())
                    self.optimizer.step()

                epoch_loss += loss.item()
                pbar.update(images.shape[0])

            train_loss = epoch_loss / len(self.train_loader)

            if self.config["use_wandb"]:
                self.accelerator.log({
                    "train_loss": train_loss,
                    "epoch": epoch,
                    "fold": fold,
                })
            if (((epoch + 1) % self.config["val_every"] == 0)
                    or epoch == 0) and self.config["no_folds"] > 1:
                val_loss, metrics_df = self.evaluate(epoch=epoch, fold=fold)

                metrics_df.to_csv(
                    f"{self.run_path}/fold_{fold}_epoch_{epoch}_metrics.csv",
                    index=False,
                )
                if self.config["use_wandb"]:
                    # Log global metrics to wandb
                    global_metrics = self.compute_global_metrics(metrics_df)
                    wandb_log = {
                        "val_loss": val_loss,
                        "epoch": epoch,
                        "fold": fold,
                        **global_metrics
                    }
                    self.accelerator.log(wandb_log)

            else:
                print(f"train loss: {train_loss}")

            self.scheduler.step(epoch)
            print("New learning rate:", self.scheduler.get_last_lr())

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
            disable=not self.accelerator.is_local_main_process,
        )
        
        # Initialize metric computer
        metric_computer = GlobalPrecisionRecallF1IoU(threshold=0.5)
        all_metrics_data = []
        
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

            loss = self.criterion(all_masks_pred, all_masks_true.float(),
                                  all_masks_weights.squeeze(1))
            epoch_loss += loss.item()

            # Get raw counts per patch
            counts = metric_computer(all_masks_true.float(), all_masks_pred, all_masks_weights.float())
            
            # Create batch metrics data
            batch_data = {
                'tp': counts['tp'].cpu().numpy(),
                'fp': counts['fp'].cpu().numpy(), 
                'fn': counts['fn'].cpu().numpy(),
                'tn': counts['tn'].cpu().numpy(),
                'register_index': all_indexes.cpu().numpy(),
                'epoch': epoch
            }
            all_metrics_data.append(batch_data)
            pbar.update(images.shape[0])

        # Combine all batch data into single arrays
        combined_data = {
            'tp': np.concatenate([batch['tp'] for batch in all_metrics_data]),
            'fp': np.concatenate([batch['fp'] for batch in all_metrics_data]),
            'fn': np.concatenate([batch['fn'] for batch in all_metrics_data]),
            'tn': np.concatenate([batch['tn'] for batch in all_metrics_data]),
            'register_index': np.concatenate([batch['register_index'] for batch in all_metrics_data]),
            'epoch': np.full(sum(len(batch['tp']) for batch in all_metrics_data), epoch)
        }
        
        # Compute per-patch metrics
        metrics_df = self.compute_per_patch_metrics(combined_data)
        
        return epoch_loss / len(self.val_loader), metrics_df

    def compute_per_patch_metrics(self, data):
        """Compute per-patch metrics from raw counts"""
        smooth = 1e-8
        
        tp = data['tp']
        fp = data['fp']
        fn = data['fn']
        tn = data['tn']
        
        # Compute per-patch metrics
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        f1 = 2 * precision * recall / (precision + recall + smooth)
        iou = tp / (tp + fp + fn + smooth)
        accuracy = (tp + tn) / (tp + fp + fn + tn + smooth)
        specificity = tn / (tn + fp + smooth)
        
        # Create old-style column names for backward compatibility
        # Only threshold 0.5 matters, so we create single columns with _0.5 suffix
        metrics_df = pd.DataFrame({
            # Raw counts (new)
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            
            # New computed metrics
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy,
            'specificity': specificity,
            
            # Old-style column names for backward compatibility (threshold 0.5 only)
            'precision_0.5': precision,
            'recall_0.5': recall,
            'f1_0.5': f1,
            'iou_0.5': iou,
            
            # Metadata
            'register_index': data['register_index'],
            'epoch': data['epoch']
        })
        
        return metrics_df
    
    def compute_global_metrics(self, metrics_df):
        """Compute global metrics by summing all TP, FP, FN, TN across patches"""
        smooth = 1e-8
        
        # Sum counts across all patches
        total_tp = metrics_df['tp'].sum()
        total_fp = metrics_df['fp'].sum()
        total_fn = metrics_df['fn'].sum()
        total_tn = metrics_df['tn'].sum()
        
        # Compute global metrics
        global_precision = total_tp / (total_tp + total_fp + smooth)
        global_recall = total_tp / (total_tp + total_fn + smooth)
        global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + smooth)
        global_iou = total_tp / (total_tp + total_fp + total_fn + smooth)
        global_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + smooth)
        global_specificity = total_tn / (total_tn + total_fp + smooth)
        
        return {
            'global_precision': global_precision,
            'global_recall': global_recall,
            'global_f1': global_f1,
            'global_iou': global_iou,
            'global_accuracy': global_accuracy,
            'global_specificity': global_specificity,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_tn': total_tn
        }

    def save_checkpoint(self, fold: int, epoch: int):
        checkpoint_name = f"{self.run_path}/fold_{fold}_epoch_{epoch}"
        self.accelerator.save_state(output_dir=checkpoint_name)
        self.accelerator.wait_for_everyone()

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

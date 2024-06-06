from collections import defaultdict
import pandas as pd
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        intersection = (logits * targets).sum(dim=(2, 3))
        logits_sum = logits.sum(dim=(2, 3))
        targets_sum = targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (
            logits_sum + targets_sum + self.smooth
        )
        dice_loss = 1 - dice_score.mean()
        return dice_loss


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=1.0, bce_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class GroupedConfusion(nn.Module):
    def __init__(
        self,
        threshold=0.5,
        smooth=1e-6,
    ):
        super(GroupedConfusion, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
        self.counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    def forward(self, logits, targets, metas=None):
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        true_positives = torch.sum((preds == 1) & (targets == 1), dim=(1, 2)).float()
        false_positives = torch.sum((preds == 1) & (targets == 0), dim=(1, 2)).float()
        false_negatives = torch.sum((preds == 0) & (targets == 1), dim=(1, 2)).float()

        for index, (tp, fp, fn) in enumerate(
            zip(
                true_positives,
                false_positives,
                false_negatives,
            )
        ):
            group = (
                metas["biome"][index].item(),
                metas["resolution_bin"][index].item(),
            )
            self.counts[group]["tp"] += tp.item()
            self.counts[group]["fp"] += fp.item()
            self.counts[group]["fn"] += fn.item()

    def compute_metrics(self, fold: int, epoch: int):
        metrics = defaultdict(lambda: {"precision": 0, "recall": 0, "f1": 0})
        for group, counts in self.counts.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = (tp + self.smooth) / (tp + fp + self.smooth)
            recall = (tp + self.smooth) / (tp + fn + self.smooth)
            f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
            metrics[group]["precision"] = precision
            metrics[group]["recall"] = recall
            metrics[group]["f1"] = f1

        records = []
        for group, metric in metrics.items():
            record = {
                "fold": fold,
                "epoch": epoch,
                "biome": group[0],
                "resolution_bin": group[1],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
            }
            records.append(record)

        metrics_df = pd.DataFrame.from_records(records)
        return metrics_df

from collections import defaultdict
import pandas as pd
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, weights):
        logits = torch.sigmoid(logits)
        intersection = (logits * targets * weights).sum(dim=(1, 2))
        logits_sum = (logits * weights).sum(dim=(1, 2))
        targets_sum = (targets * weights).sum(dim=(1, 2))
        dice_score = (2.0 * intersection + self.smooth) / (
            logits_sum + targets_sum + self.smooth
        )
        dice_loss = 1 - dice_score.mean()
        return dice_loss


class BCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, targets, weights):
        bce_loss = self.bce(logits, targets)
        weighted_bce_loss = (bce_loss * weights).mean()
        return weighted_bce_loss


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=1.0, bce_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight

    def forward(self, logits, targets, weights):
        # Calculate the weighted BCE loss
        weighted_bce_loss = self.bce(logits, targets, weights)

        # Calculate the Dice loss with weights
        dice_loss = self.dice(logits, targets, weights)

        # Combine the weighted BCE and Dice losses
        total_loss = (
            self.bce_weight * weighted_bce_loss + (1 - self.bce_weight) * dice_loss
        )
        return total_loss


class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
        """
        Initialize Tversky Focal Loss with weights.

        Parameters:
        - alpha: weight for false positives (default=0.5).
        - beta: weight for false negatives (default=0.5).
        - gamma: focusing parameter (default=1.0).
        - smooth: smoothing factor to avoid division by zero (default=1e-6).
        """
        super(TverskyFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets, weights):
        # Apply sigmoid to logits to get probabilities
        inputs = torch.sigmoid(inputs)
        # Flatten label, prediction, and weights tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weights = weights.view(-1)
        # Apply weights to the inputs and targets
        weighted_inputs = inputs * weights
        weighted_targets = targets * weights
        # Calculate true positives, false positives, and false negatives
        TP = (weighted_inputs * weighted_targets).sum()
        FP = ((1 - weighted_targets) * weighted_inputs).sum()
        FN = (weighted_targets * (1 - weighted_inputs)).sum()
        # Calculate Tversky index with weights
        Tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        # Calculate Tversky Focal Loss
        Tversky_focal_loss = (1 - Tversky_index) ** self.gamma
        return Tversky_focal_loss


class PrecisionRecallF1IoU(nn.Module):
    def __init__(self, thresholds=None, smooth=1e-8):
        super(PrecisionRecallF1IoU, self).__init__()
        if thresholds is None:
            thresholds = torch.arange(0.1, 1.0, 0.1)
        self.thresholds = thresholds  # List of thresholds to evaluate
        self.smooth = smooth  # Small constant to prevent division by zero

    def forward(self, true_masks, predicted_masks, weight_masks):
        batch_size = true_masks.size(0)
        num_thresholds = len(self.thresholds)

        # Prepare tensors to hold precision, recall, and F1 scores
        precision = torch.zeros(batch_size, num_thresholds)
        recall = torch.zeros(batch_size, num_thresholds)
        f1 = torch.zeros(batch_size, num_thresholds)
        iou = torch.zeros(batch_size, num_thresholds)

        # Iterate over each threshold
        for i, threshold in enumerate(self.thresholds):
            # Binarize the masks based on the current threshold
            true_masks_bin = (true_masks > threshold).float()
            predicted_masks_bin = (predicted_masks > threshold).float()
            weight_masks_bin = (weight_masks > threshold).float()

            # Element-wise multiplication to consider only weighted areas
            true_masks_weighted = true_masks_bin * weight_masks_bin
            predicted_masks_weighted = predicted_masks_bin * weight_masks_bin

            # Compute true positives, false positives, false negatives
            tp = (predicted_masks_weighted * true_masks_weighted).sum(dim=(1, 2))
            fp = (predicted_masks_weighted * (1 - true_masks_weighted)).sum(dim=(1, 2))
            fn = ((1 - predicted_masks_weighted) * true_masks_weighted).sum(dim=(1, 2))

            # Calculate precision, recall, and F1 score
            precision[:, i] = tp / (tp + fp + self.smooth)
            recall[:, i] = tp / (tp + fn + self.smooth)
            f1[:, i] = (
                2
                * precision[:, i]
                * recall[:, i]
                / (precision[:, i] + recall[:, i] + self.smooth)
            )
            iou[:, i] = tp / (tp + fp + fn + self.smooth)

        return precision, recall, f1, iou

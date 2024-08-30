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


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=1.0, bce_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight

    def forward(self, logits, targets, weights):
        # Apply the BCEWithLogitsLoss
        bce_loss = self.bce(logits, targets)
        # Multiply the loss by the weights and reduce
        weighted_bce_loss = (bce_loss * weights).mean()

        # Calculate the Dice loss with weights
        dice_loss = self.dice(logits, targets, weights)

        # Combine the weighted BCE and Dice losses
        total_loss = (
            self.bce_weight * weighted_bce_loss + (1 - self.bce_weight) * dice_loss
        )
        return total_loss
    

class PrecisionRecallF1(nn.Module):
    def __init__(self, thresholds=None, smooth=1e-8):
        super(PrecisionRecallF1, self).__init__()
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
            f1[:, i] = 2 * precision[:, i] * recall[:, i] / (precision[:, i] + recall[:, i] + self.smooth)

        return precision, recall, f1


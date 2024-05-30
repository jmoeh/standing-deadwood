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


class Confusion(nn.Module):
    def __init__(self, threshold=0.5, smooth=1e-6):
        super(Confusion, self).__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        true_positives = torch.sum((preds == 1) & (targets == 1), dim=(2, 3)).float()
        false_positives = torch.sum((preds == 1) & (targets == 0), dim=(2, 3)).float()
        false_negatives = torch.sum((preds == 0) & (targets == 1), dim=(2, 3)).float()

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        return precision.mean(), recall.mean(), f1.mean()

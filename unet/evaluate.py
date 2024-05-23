import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet.dice_score import dice_loss, confusion_values


@torch.inference_mode()
def evaluate(net, criterion, dataloader: DataLoader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0
    val_rows = []
    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for images, masks_true, images_meta in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation round",
            unit="batch",
            leave=False,
        ):
            # move images and labels to correct device and type
            images = images.to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            masks_true = masks_true.to(device=device, dtype=torch.long).squeeze()
            # predict the mask
            masks_pred = net(images).squeeze()
            assert (
                masks_true.min() >= 0 and masks_true.max() <= 1
            ), "True mask indices should be in [0, 1]"

            for tile_index in range(len(images)):
                mask_pred = masks_pred[tile_index].squeeze(1)
                mask_true = masks_true[tile_index].squeeze(1).float()

                tile_ce_loss = criterion(mask_pred, mask_true)
                tile_dice_loss = dice_loss(
                    F.sigmoid(mask_pred), mask_true, reduce_batch_first=False
                )
                tile_loss = tile_ce_loss + tile_dice_loss

                for threshold in np.arange(0.1, 0.9, 0.1):
                    mask_pred_treshold = (masks_pred[tile_index] > threshold).float()
                    precision, recall, f1 = confusion_values(
                        mask_pred_treshold, mask_true
                    )
                    val_row = {
                        "treshold": threshold,
                        "precision": precision.item(),
                        "recall": recall.item(),
                        "f1": f1.item(),
                        "ce_loss": tile_ce_loss.item(),
                        "dice_loss": tile_dice_loss.item(),
                        "biome": images_meta["biome"][tile_index].item(),
                        "resolution_bin": images_meta["resolution_bin"][
                            tile_index
                        ].item(),
                    }
                    val_rows.append(val_row)
                val_loss += tile_loss.item()

    net.train()
    return val_loss / (len(dataloader) * dataloader.batch_size), pd.DataFrame(val_rows)

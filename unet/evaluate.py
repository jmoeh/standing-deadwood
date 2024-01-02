import torch
import torch.nn.functional as F
from tqdm import tqdm

from unet.dice_score import dice_coeff


@torch.inference_mode()
def evaluate(net, criterion, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    dice_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for image, mask_true in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation round",
            unit="batch",
            leave=False,
        ):
            # move images and labels to correct device and type
            image = image.to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            assert (
                mask_true.min() >= 0 and mask_true.max() <= 1
            ), "True mask indices should be in [0, 1]"

            # apply sigmoid to the mask
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            # compute the Dice loss
            item_loss = criterion(mask_pred, mask_true.float())
            item_loss += 1 - dice_score

            dice_loss += item_loss.item()

    net.train()
    return dice_score / max(num_val_batches, 1), dice_loss / max(num_val_batches, 1)

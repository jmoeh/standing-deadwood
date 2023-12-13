import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DeadwoodDataset(Dataset):
    def __getitem__(self, index):
        image_name = self.images_paths[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        mask = Image.open(
            os.path.join(self.mask_dir, image_name.replace("8857", "8857_mask", 1))
        ).convert("L")

        image_tensor = transforms.ToTensor()(image).float().contiguous()
        mask_tensor = transforms.PILToTensor()(mask).squeeze(1).long().contiguous()

        return image_tensor, mask_tensor

    def __init__(self, image_dir, mask_dir):
        super(DeadwoodDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images_paths = os.listdir(image_dir)

    def __len__(self):
        return len(self.images_paths)

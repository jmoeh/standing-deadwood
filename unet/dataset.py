import os
from PIL import Image
from torch.utils.data import Dataset


class DeadwoodDataset(Dataset):
    def __getitem__(self, index):
        image_name = self.images_paths[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        mask = Image.open(
            os.path.join(self.mask_dir, image_name.replace(".tif", "_mask.tif"))
        ).convert("L")

        image = self.transforms_image(image)
        mask = self.transforms_mask(mask)

        return image, mask

    def __init__(self, image_dir, mask_dir, transforms_image, transforms_mask):
        super(DeadwoodDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms_image = transforms_image
        self.transforms_mask = transforms_mask

        self.images_paths = os.listdir(image_dir)

    def __len__(self):
        return len(self.images_paths)

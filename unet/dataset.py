from collections import defaultdict
import random

import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split


class DeadwoodDataset(Dataset):
    def __init__(
        self,
        register_df,
        test_size=0.3,
        random_seed=10,
    ):
        super(DeadwoodDataset, self).__init__()
        self.register_df = register_df
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        train_register_df, test_register_df = train_test_split(
            register_df,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=register_df[["biome", "resolution_bin"]],
        )

        self.train_register_df = train_register_df
        self.test_register_df = test_register_df

        # get indices of rows from original dataframe
        self.train_indices = self.train_register_df.index.tolist()
        self.test_indices = self.test_register_df.index.tolist()

    def get_train_test(self):
        return Subset(self, self.train_indices), Subset(self, self.test_indices)

    def __getitem__(self, index):
        image_path = self.register_df.iloc[index]["file_path"]
        image = Image.open(image_path).convert("RGB")
        mask_path = image_path.replace(".tif", "_mask.tif")
        mask = Image.open(mask_path).convert("L")
        mutual_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        image_transforms = transforms.Compose(
            [
                transforms.RandomAutocontrast(p=0.2),
            ]
        )
        image_tensor, mask_tensor = RandomTransform(mutual_transforms)(
            image_transforms(image), mask
        )
        image_tensor = transforms.ToTensor()(image_tensor).float().contiguous()
        mask_tensor = (
            transforms.PILToTensor()(mask_tensor).squeeze(1).long().contiguous()
        )
        return image_tensor, mask_tensor, self.register_df.iloc[index].to_dict()

    def __len__(self):
        return len(self.register_df)


class RandomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        seed = torch.randint(0, 2**32, (1,))
        random_state = torch.random.fork_rng([seed.item()])

        transformed_image = self.transform(image)

        torch.random.set_rng_state(random_state)
        transformed_mask = self.transform(mask)

        torch.random.set_rng_state(random_state)
        return transformed_image, transformed_mask

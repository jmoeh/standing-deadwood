from collections import defaultdict
import random

import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split


class DeadwoodDataset(Dataset):
    def __init__(
        self,
        register_df,
        no_folds=5,
        random_seed=10,
    ):
        super(DeadwoodDataset, self).__init__()
        self.register_df = register_df
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # split the dataset into train and test but grouped by base_image_ids so that the same base image is not in both train and test
        # but stratified by biome and resolution_bin
        self.base_file_register = (
            register_df.groupby("base_file_name").agg({"biome": "first"}).reset_index()
        )
        kfold = StratifiedKFold(
            n_splits=no_folds, random_state=self.random_seed, shuffle=True
        )
        self.folds = kfold.split(
            self.base_file_register, self.base_file_register["biome"].astype(str)
        )

        # get indices per fold of register_df
        self.train_indices = []
        self.test_indices = []

        for train_index, test_index in self.folds:
            train_files = self.base_file_register.iloc[train_index]["base_file_name"]
            test_files = self.base_file_register.iloc[test_index]["base_file_name"]

            train_indices = self.register_df[
                self.register_df["base_file_name"].isin(train_files)
            ].index.tolist()
            test_indices = self.register_df[
                self.register_df["base_file_name"].isin(test_files)
            ].index.tolist()

            self.train_indices.append(train_indices)
            self.test_indices.append(test_indices)

    def get_fold(self, fold):
        return Subset(self, self.train_indices[fold]), Subset(
            self, self.test_indices[fold][:64]
        )

    def get_train_sample_weights(self, fold, balancing_factor=0.5):
        train_register = self.register_df.iloc[self.train_indices[fold]]
        value_counts = train_register["resolution_bin"].value_counts()

        # apply balancing factor to the counts before inverting to get weights
        sqrt_counts = value_counts.apply(lambda x: x**balancing_factor)
        weights = 1 / sqrt_counts

        return train_register["resolution_bin"].map(weights)

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
        image_transforms = transforms.Compose([transforms.RandomAutocontrast()])
        image_tensor, mask_tensor = RandomTransform(mutual_transforms)(image, mask)
        image_tensor = (
            transforms.ToTensor()(image_transforms(image_tensor)).float().contiguous()
        )
        mask_tensor = (
            transforms.PILToTensor()(mask_tensor).squeeze(0).long().contiguous()
        )

        return image_tensor, mask_tensor, self.register_df.iloc[index].to_dict()

    def __len__(self):
        return len(self.register_df)


class RandomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        seed = torch.randint(0, 2**32, (1,))
        torch.manual_seed(seed.item())

        transformed_image = self.transform(image)

        torch.manual_seed(seed.item())
        transformed_mask = self.transform(mask)

        return transformed_image, transformed_mask

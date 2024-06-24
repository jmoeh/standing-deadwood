import random

import numpy as np
import torch
from PIL import Image
from rasterio import windows
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms


class DeadwoodDataset(Dataset):
    def __init__(
        self,
        register_df,
        images_dir,
        no_folds=5,
        random_seed=1,
        test_size=0.2,
        bins=np.arange(0, 0.21, 0.02),
        nodata_value=255,
    ):
        super(DeadwoodDataset, self).__init__()
        self.register_df = register_df
        self.random_seed = random_seed
        self.images_dir = images_dir
        self.nodata_value = nodata_value
        self.no_folds = no_folds

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # split the dataset into train and test but grouped by base_image_ids so that the same base image is not in both train and test
        # but stratified by biome and resolution_bin
        self.base_file_register = (
            register_df.groupby("base_file_name")
            .agg({"biome": "first", "resolution_bin": "min"})
            .reset_index()
        )

        if test_size > 0:
            self.base_train_val_register, self.base_test_register = train_test_split(
                self.base_file_register,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=self.base_file_register[["biome"]],
            )

            test_files = self.base_test_register["base_file_name"]
            self.test_indices = self.register_df[
                self.register_df["base_file_name"].isin(test_files)
            ].index.tolist()
        else:
            self.base_train_val_register = self.base_file_register

        self.base_train_val_register = self.base_train_val_register[
            self.base_train_val_register["resolution_bin"].isin(bins)
        ]

        # get indices per fold of register_df
        self.train_indices = []
        self.val_indices = []

        if self.no_folds == 1:
            train_files = self.base_train_val_register["base_file_name"]
            train_indices = self.register_df[
                self.register_df["base_file_name"].isin(train_files)
            ].index.tolist()
            self.train_indices.append(train_indices)
            self.val_indices.append([])
            return
        elif self.no_folds > 1:
            kfold = StratifiedKFold(
                n_splits=self.no_folds, random_state=self.random_seed, shuffle=True
            )
            self.folds = kfold.split(
                self.base_train_val_register,
                self.base_train_val_register["biome"].astype(str),
            )

            for train_index, val_index in self.folds:
                train_files = self.base_file_register.iloc[train_index][
                    "base_file_name"
                ]
                val_files = self.base_file_register.iloc[val_index]["base_file_name"]
                train_indices = self.register_df[
                    self.register_df["base_file_name"].isin(train_files)
                ].index.tolist()
                val_indices = self.register_df[
                    self.register_df["base_file_name"].isin(val_files)
                ].index.tolist()

                self.train_indices.append(train_indices)
                self.val_indices.append(val_indices)

    def get_train_set(self):
        return Subset(self, self.train_indices)

    def get_val_set(self):
        return Subset(self, self.val_indices)

    def get_test_set(self):
        return Subset(self, self.test_indices)

    def get_train_val_fold(self, fold):
        if self.no_folds == 1:
            raise ValueError("fold index out of range")
        return Subset(self, self.train_indices[fold]), Subset(
            self, self.val_indices[fold]
        )

    def get_train_sample_weights(self, fold, balancing_factor=0.5):
        train_register = self.register_df.iloc[self.train_indices[fold]]
        value_counts = train_register["resolution_bin"].value_counts()

        # apply balancing factor to the counts before inverting to get weights
        sqrt_counts = value_counts.apply(lambda x: x**balancing_factor)
        weights = 1 / sqrt_counts

        return torch.from_numpy(train_register["resolution_bin"].map(weights).values)

    def __getitem__(self, index):
        while True:
            image_path = (
                f'{self.images_dir}{self.register_df.iloc[index]["global_file_path"]}'
            )
            try:
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
                        transforms.RandomAutocontrast(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
                mask_transforms = transforms.Compose(
                    [
                        transforms.PILToTensor(),
                    ]
                )
                image_tensor, mask_tensor = CoupledRandomTransform(mutual_transforms)(
                    image, mask
                )
                image_tensor = image_transforms(image_tensor).float().contiguous()
                mask_tensor = (
                    mask_transforms(mask_tensor).squeeze(0).long().contiguous()
                )

                weight_tensor = torch.ones_like(mask_tensor, dtype=torch.float32)
                weight_tensor[mask_tensor == self.nodata_value] = (
                    0  # Assign weight 0 to 'nodata' areas
                )

                return (
                    image_tensor,
                    mask_tensor,
                    weight_tensor,
                    self.register_df.iloc[index].to_dict(),
                )
            except:
                print(f"skipping image {image_path}")
                index = (index + 1) % len(self.register_df)

    def __len__(self):
        return len(self.register_df)


class CoupledRandomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask):
        seed = torch.randint(0, 2**32, (1,))
        torch.manual_seed(seed.item())

        transformed_image = self.transform(image)

        torch.manual_seed(seed.item())
        transformed_mask = self.transform(mask)

        return transformed_image, transformed_mask


def get_windows(xmin, ymin, xmax, ymax, tile_width, tile_height, overlap):
    xstep = tile_width - overlap
    ystep = tile_height - overlap
    for x in range(xmin, xmax, xstep):
        if x + tile_width > xmax:
            x = xmax - tile_width
        for y in range(ymin, ymax, ystep):
            if y + tile_height > ymax:
                y = ymax - tile_height
            window = windows.Window(x, y, tile_width, tile_height)
            yield window

from collections import defaultdict
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import KFold


class DeadwoodDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        n_folds=10,
        random_seed=1,
    ):
        super(DeadwoodDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # group the tiles by the base images and perform kfold split on the base images
        self.images_paths = os.listdir(image_dir)
        self.file_groups = defaultdict(list)
        base_file_pattern = r".+?(?=_\d+_\d+\.tif)"

        for index, image_path in enumerate(self.images_paths):
            base_file_name = re.search(base_file_pattern, image_path).group(0)
            self.file_groups[base_file_name].append(index)

        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        self.folds = list(self.kfold.split(list(self.file_groups.keys())))

    """Returns the train and test subsets for the given fold index"""

    def get_fold(self, fold_idx):
        train_indices, test_indices = self.folds[fold_idx]
        train_files = [
            idx
            for group_idx in train_indices
            for idx in self.file_groups[list(self.file_groups.keys())[group_idx]]
        ]
        test_files = [
            idx
            for group_idx in test_indices
            for idx in self.file_groups[list(self.file_groups.keys())[group_idx]]
        ]
        train_subset = torch.utils.data.Subset(self, train_files)
        test_subset = torch.utils.data.Subset(self, test_files)

        return train_subset, test_subset

    def __getitem__(self, index):
        image_name = self.images_paths[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")

        pattern = re.compile(r"^(.+?)_(\d+)_(\d+)\.tif$")
        match = pattern.match(image_name)

        if match:
            mask_name = f"{match.group(1)}_mask_{match.group(2)}_{match.group(3)}.tif"
            mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")

            image_tensor = transforms.ToTensor()(image).float().contiguous()
            mask_tensor = transforms.PILToTensor()(mask).squeeze(1).long().contiguous()

            return image_tensor, mask_tensor
        else:
            raise ValueError(f"Invalid tile name: {image_name}")

    def __len__(self):
        return len(self.images_paths)

from collections import defaultdict
import random

import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import KFold


class DeadwoodDataset(Dataset):
    def __init__(
        self,
        register_file,
        n_folds=10,
        random_seed=1,
    ):
        super(DeadwoodDataset, self).__init__()
        self.register_file = register_file
        self.register_df = pd.read_csv(register_file)
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        
        bins = np.arange(0.00, 0.22, 0.02) # increase stop value to 0.22 to include the last bin
        
        # group the tiles by the base images and perform kfold split on the base images
        self.file_groups = defaultdict(lambda: defaultdict(list))
        for index, register_row in self.register_df.iterrows():
            base_file_name = register_row["base_file_name"]
            resolution = register_row["resolution"]
            bin_index = np.digitize(resolution, bins)
            self.file_groups[base_file_name][bins[bin_index - 1]].append(index)

        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        self.folds = list(self.kfold.split(list(self.file_groups.keys())))

    """Returns the train and test subsets for the given fold index"""

    def get_fold(self, fold_idx, epochs=1, n_res_samples=1):
        train_indices, test_indices = self.folds[fold_idx]
        
        train_files_epochs = []
        resolution_groups = defaultdict(list)
        for group_idx in train_indices:
            for resolution, resolution_indices in self.file_groups[list(self.file_groups.keys())[group_idx]].items():
                resolution_groups[resolution].extend(resolution_indices)
        for _ in range(epochs):
            train_files = []
            for resolution, resolution_indices in resolution_groups.items():
                sampled_indices = np.random.choice(resolution_indices, size=n_res_samples, replace=False)
                train_files.extend(sampled_indices)
            train_files_epochs.append(train_files)
            
        test_files = [
            idx
            for group_idx in test_indices
            for resolution in self.file_groups[list(self.file_groups.keys())[group_idx]].values()
            for idx in resolution
        ]
        return train_files_epochs, test_files

    def __getitem__(self, index):
        image_path = self.register_df.iloc[index]["file_path"]
        image = Image.open(image_path).convert("RGB")
        
        mask_path = image_path.replace(".tif", "_mask.tif")
        mask = Image.open(mask_path).convert("L")

        image_tensor = transforms.ToTensor()(image).float().contiguous()
        mask_tensor = transforms.PILToTensor()(mask).squeeze(1).long().contiguous()

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.register_df)

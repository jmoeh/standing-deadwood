from torch.utils.data import Sampler
from numpy.random import shuffle, default_rng


class CappedSampler(Sampler):

    def __init__(
        self,
        dataset,
        fold,
        max_samples_per_ortho=50,
        max_samples_per_3D_class=100,
        oversample=True,
    ):

        self.dataset = dataset
        self.max_samples_per_ortho = max_samples_per_ortho
        self.max_samples_per_3D_class = max_samples_per_3D_class
        self.oversample = oversample
        self.resample_columns = [
            "resolution_bin", "biome_group", "mask_filled"
        ]

        # create subset of dataset based on fold
        self.register_subset = dataset.register_df.copy()
        self.register_subset = self.register_subset.iloc[
            dataset.train_indices[fold]]

        self.indices = None

        self.rng = default_rng()

        self.select_indices()

    def __len__(self):
        return len(self.indices)

    def select_indices(self):

        temp_df = self.register_subset.copy()

        # remember orignal indices
        temp_df["orig_index"] = temp_df.index

        # for each orthophoto and classes combination select at most 50 pixels
        temp_df = temp_df.groupby(
            self.resample_columns + ["base_file_name"], observed=False).apply(
                lambda sdf: sdf.sample(frac=1, random_state=self.rng).head(
                    self.max_samples_per_ortho)).reset_index(drop=True)

        if self.oversample:
            # for each biome - deadwood_bin combination select exactly 100 pixels
            temp_df = temp_df.groupby(
                self.resample_columns, observed=False).apply(
                    lambda sdf: sdf.sample(n=self.max_samples_per_3D_class,
                                           replace=True,
                                           random_state=self.rng)).reset_index(
                                               drop=True)

        else:

            # for each biome - deadwood_bin combination select at most 100 pixels
            temp_df = temp_df.groupby(
                self.resample_columns, observed=False).apply(
                    lambda sdf: sdf.sample(frac=1, random_state=self.rng).head(
                        self.max_samples_per_3D_class)).reset_index(drop=True)

        # extract indices
        self.indices = temp_df["orig_index"].values

        # print distribution nicely
        print("CappedSampler: ")
        print(
            temp_df.groupby([
                "resolution_bin", "biome_group"
            ]).size().to_frame().reset_index().pivot(index="resolution_bin",
                                                     columns="biome_group"))

        # inplace
        shuffle(self.indices)

    def __iter__(self):
        self.select_indices()

        return iter(self.indices)

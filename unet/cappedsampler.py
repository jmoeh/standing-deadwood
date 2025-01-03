from torch.utils.data import Sampler
from numpy.random import shuffle, default_rng


class CappedSampler(Sampler):

    def __init__(
        self,
        dataset,
        max_samples_per_ortho=50,
        max_samples_per_3D_class=100,
    ):

        self.dataset = dataset
        self.max_samples_per_ortho = max_samples_per_ortho
        self.max_samples_per_3D_class = max_samples_per_3D_class
        self.resample_columns = ["resolution_bin", "biome", "mask_filled"]

        self.indices = None

        self.rng = default_rng()

    def select_indices(self):

        temp_df = self.dataset.register_df.copy()

        # remember orignal indices
        temp_df["orig_index"] = temp_df.index

        # for each orthophoto and classes combination select at most 50 pixels
        temp_df = temp_df.groupby(
            self.resample_columns + ["filename"], observed=False).apply(
                lambda sdf: sdf.sample(frac=1, random_state=self.rng).head(
                    self.max_samples_per_ortho)).reset_index(drop=True)

        # for each biome - deadwood_bin combination select at most 100 pixels
        temp_df = temp_df.groupby(self.resample_columns, observed=False).apply(
            lambda sdf: sdf.sample(frac=1, random_state=self.rng).head(
                self.max_samples_per_3D_class)).reset_index(drop=True)

        # extract indices
        self.indices = temp_df["orig_index"].values

        # print distribution nicely
        print("CappedSampler: ")
        print(
            temp_df.groupby(
                self.resample_columns).size().to_frame().reset_index().pivot(
                    index="deadwood_bin", columns="biome"))

        # inplace
        shuffle(self.indices)

    def __iter__(self):
        self.select_indices()

        return iter(self.indices)

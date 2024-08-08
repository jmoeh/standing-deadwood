import os
import argparse

import pandas as pd
import rasterio
from tqdm import tqdm


def execute(workspace, images_dir):
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    meta_df = pd.DataFrame(
        columns=[
            "filename",
            "west",
            "east",
            "south",
            "north",
            "width",
            "height",
            "crs",
        ]
    )
    for filename in tqdm(os.listdir(images_dir)):
        with rasterio.open(os.path.join(images_dir, filename)) as src:
            meta_df = pd.concat(
                [
                    meta_df,
                    pd.DataFrame(
                        {
                            "filename": [filename],
                            "west": [src.bounds.left],
                            "east": [src.bounds.right],
                            "south": [src.bounds.bottom],
                            "north": [src.bounds.top],
                            "width": [src.width],
                            "height": [src.height],
                            "crs": [
                                src.crs.to_string()
                            ],  # Convert CRS object to string
                        }
                    ),
                ],
                ignore_index=True,
                axis=0,
            )
    meta_df.to_csv(f"{workspace}/images_meta.csv", index=False)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Extend Metadata",
        description="Extend metadata with image dimensions",
    )
    arg_parser.add_argument("--workspace", type=str, required=True)
    arg_parser.add_argument("--images_dir", type=str, required=True)
    args = arg_parser.parse_args()

    execute(args.workspace, args.images_dir)

import argparse
import os

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from shapely.geometry import mapping
from tqdm import tqdm

# Add argument parser to get label dir, metadata file, and output dir
parser = argparse.ArgumentParser(description="Generate orthophoto masks.")
parser.add_argument(
    "labels_dir",
    help="Path to the directory containing the GeoPackage label files",
)
parser.add_argument(
    "-m",
    "--metadata_file",
    help="Path to the metadata file",
    required=True,
)
parser.add_argument(
    "-o",
    "--out_dir",
    help="Path where the masks will be saved",
    default="./",
    required=False,
)
args = parser.parse_args()

# add new filename map column to find metadata for each image
df_meta = pd.read_csv(args.metadata_file)
df_meta["filename_map"] = df_meta["filename"].str.replace("_ortho.tif", "")

# assert there is metadata for all gpkg files in given directory
for filename in os.listdir(args.labels_dir):
    assert filename.endswith(".gpkg")
    assert (
        filename.replace("_ortho_polygons.gpkg", "") in df_meta["filename_map"].values
    )

# iterate over all gpkg files in given directory
for filename in tqdm(os.listdir(args.labels_dir)):
    if filename.endswith(".gpkg"):
        filepath = os.path.join(args.labels_dir, filename)

        # Get metadata for current gpkg file
        filename_map = filename.replace("_ortho_polygons.gpkg", "")
        file_meta = df_meta.loc[df_meta["filename_map"] == filename_map].to_dict(
            "records"
        )[0]
        resolution = file_meta["width"] / abs(file_meta["right"] - file_meta["left"])

        # crop the image based on the extend of the aoi layer
        gdf_aoi = gpd.read_file(filepath, layer="aoi")
        aoi_bounds = gdf_aoi.total_bounds
        out_image_height = int(abs(aoi_bounds[1] - aoi_bounds[3]) * resolution)
        out_image_width = int(abs(aoi_bounds[0] - aoi_bounds[2]) * resolution)

        out_image = np.zeros((out_image_height, out_image_width), dtype=np.uint8)
        transform = from_bounds(
            *aoi_bounds, width=out_image_width, height=out_image_height
        )

        # Read in gpkg file and determine if standing deadwood is present
        layers = fiona.listlayers(filepath)
        if "standing_deadwood" in layers:
            gdf_label = gpd.read_file(filepath, layer="standing_deadwood")
            # Rasterize polygons
            for _, row in gdf_label.iterrows():
                geom = mapping(row["geometry"])
                mask = geometry_mask(
                    [geom], transform=transform, invert=True, out_shape=out_image.shape
                )
                out_image[mask] = 1

        # Save image
        with rasterio.open(
            os.path.join(args.out_dir, filename_map + "_ortho_mask.tif"),
            "w",
            driver="GTiff",
            height=out_image.shape[0],
            width=out_image.shape[1],
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(out_image, 1)

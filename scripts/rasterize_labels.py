import argparse
from bz2 import compress
import concurrent.futures
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


def process_file(filename, df_meta):
    if filename.endswith(".gpkg"):
        filepath = os.path.join(args.labels_dir, filename)
        # Get metadata for current gpkg file
        filename_map = filename.replace("_ortho_polygons.gpkg", "")
        file_meta = df_meta.loc[df_meta["filename_map"] == filename_map].to_dict(
            "records"
        )[0]
        out_image = np.zeros((file_meta["height"], file_meta["width"]), dtype=np.uint8)
        transform = from_bounds(
            north=file_meta["top"],
            south=file_meta["bottom"],
            west=file_meta["left"],
            east=file_meta["right"],
            width=file_meta["width"],
            height=file_meta["height"],
        )
        # Read in gpkg file and determine if standing deadwood is present
        layers = fiona.listlayers(filepath)
        if "standing_deadwood" in layers:
            gdf_label = gpd.read_file(filepath, layer="standing_deadwood")
            gdf_label = gdf_label.to_crs("EPSG:8857")
            # Rasterize polygons
            for _, row in gdf_label.iterrows():
                geom = mapping(row["geometry"])
                mask = geometry_mask(
                    [geom], transform=transform, invert=True, out_shape=out_image.shape
                )
                out_image[mask] = 1

        # Save image
        with rasterio.open(
            os.path.join(args.out_dir, filename_map + "_ortho_8857_mask.tif"),
            "w",
            driver="GTiff",
            compress="DEFLATE",
            height=out_image.shape[0],
            width=out_image.shape[1],
            count=1,
            dtype="uint8",
            crs="EPSG:8857",
            transform=transform,
        ) as dst:
            dst.write(out_image, 1)


parser = argparse.ArgumentParser(description="Rasterize labels")
parser.add_argument(
    "labels_dir",
    help="Path to the directory containing the GeoPackage label files",
)
parser.add_argument(
    "-o",
    "--out_dir",
    help="Path where the rasterized features will be stored",
    default="./",
    required=False,
)
parser.add_argument(
    "-gw",
    "--gr_path",
    help="Path to the the gdal_rasterize binary",
    default="gdal_rasterize",
    required=False,
)
parser.add_argument(
    "-m",
    "--metadata_file",
    help="Path to the metadata file",
    required=True,
)
parser.add_argument(
    "-j",
    "--jobs",
    type=int,
    default=1,
    help="Number of parallel jobs to run",
    required=False,
)
args = parser.parse_args()

# add new filename map column to find metadata for each image
df_meta = pd.read_csv(args.metadata_file)
df_meta["filename_map"] = df_meta["filename"].str.replace("_ortho_8857.tif", "")

# assert there is metadata for all gpkg files in given directory
for filename in os.listdir(args.labels_dir):
    assert filename.endswith(".gpkg")
    assert (
        filename.replace("_ortho_polygons.gpkg", "") in df_meta["filename_map"].values
    )

with tqdm(total=len(os.listdir(args.labels_dir))) as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(process_file, filename, df_meta)
            for filename in os.listdir(args.labels_dir)
        ]
        # Wait for all futures to complete
        for _ in concurrent.futures.as_completed(futures):
            pbar.update(1)

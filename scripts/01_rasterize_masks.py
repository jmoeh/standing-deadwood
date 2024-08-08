import os
import sys
import argparse

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.realpath(os.path.abspath(""))))

from utils.parallel import paral


def process_file(
    filename,
    df_meta,
):
    if filename.endswith(".gpkg"):
        if filename.replace("_polygons.gpkg", "") in df_meta["filename_map"].values:
            filepath = os.path.join(labels_dir, filename)

            # Get metadata for current gpkg file
            filename_map = filename.replace("_polygons.gpkg", "")
            out_filepath = os.path.join(out_dir, filename_map + "_mask.tif")
            file_meta = df_meta.loc[df_meta["filename_map"] == filename_map].to_dict(
                "records"
            )[0]

            # skip generation if file already exists or label quality is too low
            if os.path.exists(out_filepath):
                return
            if file_meta["label_quality"] < 2 or file_meta["has_labels"] == 0:
                return

            out_image = np.zeros(
                (file_meta["height"], file_meta["width"]), dtype=np.uint8
            )
            transform = from_bounds(
                north=file_meta["north"],
                south=file_meta["south"],
                west=file_meta["west"],
                east=file_meta["east"],
                width=file_meta["width"],
                height=file_meta["height"],
            )
            # Read in gpkg file and determine if standing deadwood is present
            layers = fiona.listlayers(filepath)
            if "standing_deadwood" in layers:
                gdf_label = gpd.read_file(filepath, layer="standing_deadwood")
                gdf_label = gdf_label.to_crs(file_meta["crs"])
                if not gdf_label.empty:
                    mask = geometry_mask(
                        gdf_label["geometry"].dropna().tolist(),
                        transform=transform,
                        invert=True,
                        out_shape=out_image.shape,
                    )
                    out_image[mask] = 1

            if "brown_trees" in layers:
                gdf_label = gpd.read_file(filepath, layer="brown_trees")
                gdf_label = gdf_label.to_crs(file_meta["crs"])
                if not gdf_label.empty:
                    # Rasterize polygons
                    mask = geometry_mask(
                        gdf_label["geometry"].dropna().tolist(),
                        transform=transform,
                        invert=True,
                        out_shape=out_image.shape,
                    )
                    out_image[mask] = 1

            if "parts" in layers:
                gdf_label = gpd.read_file(filepath, layer="parts")
                gdf_label = gdf_label.to_crs(file_meta["crs"])
                if not gdf_label.empty:
                    # Rasterize polygons
                    mask = geometry_mask(
                        gdf_label["geometry"].dropna().tolist(),
                        transform=transform,
                        invert=True,
                        out_shape=out_image.shape,
                    )
                    out_image[mask] = 1

            if "aoi" in layers:
                gdf_label = gpd.read_file(filepath, layer="aoi")
                gdf_label = gdf_label.to_crs(file_meta["crs"])
                if not gdf_label.empty:
                    # Rasterize polygons
                    mask = geometry_mask(
                        gdf_label["geometry"].dropna().tolist(),
                        transform=transform,
                        invert=False,
                        out_shape=out_image.shape,
                    )
                    out_image[mask] = 255

            # Save image
            with rasterio.open(
                out_filepath,
                "w",
                driver="GTiff",
                compress="DEFLATE",
                height=out_image.shape[0],
                width=out_image.shape[1],
                count=1,
                dtype="uint8",
                crs=file_meta["crs"],
                transform=transform,
                nodata=255,
            ) as dst:
                dst.write(out_image, 1)


def execute(workspace, labels_dir, cores):
    df_meta = pd.read_csv(f"{workspace}/images_meta.csv")
    df_meta["filename_map"] = df_meta["filename"].str.replace(".tif", "")
    output = paral(
        process_file,
        [
            os.listdir(labels_dir),
            [df_meta] * len(os.listdir(labels_dir)),
            [workspace] * len(os.listdir(labels_dir)),
        ],
        num_cores=cores,
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Extend Metadata",
        description="Extend metadata with image dimensions",
    )
    arg_parser.add_argument("--workspace", type=str, required=True)
    arg_parser.add_argument("--labels_dir", type=str, required=True)
    arg_parser.add_argument("--cores", type=int, required=True)

    args = arg_parser.parse_args()

    execute(args.workspace, args.images_dir)

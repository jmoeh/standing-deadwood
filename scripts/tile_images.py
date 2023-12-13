import argparse
import concurrent.futures
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import windows
from rasterio.features import geometry_window
from tqdm import tqdm

tile_width = 512
tile_height = 512
tile_overlap = 0
out_filename = "{}_{}_{}.tif"


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


def process_file(image_filepath):
    image_filename = os.path.basename(image_filepath)

    mask_filename = image_filename.replace("8857.tif", "8857_mask.tif")
    mask_filepath = os.path.join(args.masks_dir, mask_filename)

    label_filename = image_filename.replace("8857.tif", "polygons.gpkg")
    label_filepath = os.path.join(args.labels_dir, label_filename)
    gdf_label = gpd.read_file(label_filepath, layer="aoi")
    gdf_label = gdf_label.to_crs("EPSG:8857")

    with rasterio.open(image_filepath) as image_src:
        with rasterio.open(mask_filepath) as mask_src:
            image_metadata = image_src.meta.copy()
            mask_metadata = mask_src.meta.copy()

            for _, row in gdf_label.iterrows():
                aoi_window = geometry_window(image_src, [row.geometry])
                xmin, ymin = aoi_window.col_off, aoi_window.row_off
                xmax, ymax = xmin + aoi_window.width, ymin + aoi_window.height

                for window in get_windows(
                    xmin, ymin, xmax, ymax, tile_width, tile_height, tile_overlap
                ):
                    transform = windows.transform(window, image_src.transform)
                    image_metadata["transform"] = transform
                    image_metadata["width"], image_metadata["height"] = (
                        window.width,
                        window.height,
                    )
                    mask_metadata["transform"] = transform
                    mask_metadata["width"], mask_metadata["height"] = (
                        window.width,
                        window.height,
                    )
                    image_out_filepath = os.path.join(
                        args.image_out_dir,
                        out_filename.format(
                            image_filename.replace(".tif", ""),
                            window.col_off,
                            window.row_off,
                        ),
                    )
                    mask_out_filepath = os.path.join(
                        args.mask_out_dir,
                        out_filename.format(
                            mask_filename.replace(".tif", ""),
                            window.col_off,
                            window.row_off,
                        ),
                    )

                    out_image = image_src.read(window=window)

                    if np.count_nonzero(out_image == 255) / out_image.size < 0.01:
                        out_mask = mask_src.read(window=window)
                        with rasterio.open(
                            image_out_filepath, "w", **image_metadata
                        ) as dst:
                            dst.write(out_image)
                        with rasterio.open(
                            mask_out_filepath, "w", **mask_metadata
                        ) as dst:
                            dst.write(out_mask)


os.environ["GDAL_PAM_ENABLED"] = "NO"

parser = argparse.ArgumentParser(description="Rasterize labels")
parser.add_argument(
    "images_dir",
    help="Path to the directory containing the to be tiled images",
)
parser.add_argument(
    "masks_dir",
    help="Path to the directory containing the to be tiled masks",
)
parser.add_argument(
    "-l",
    "--labels_dir",
    help="Path to the directory containing the GeoPackage label files",
    required=True,
)
parser.add_argument(
    "-io",
    "--image_out_dir",
    help="Path where the tiled images will be stored",
    required=True,
)
parser.add_argument(
    "-mo",
    "--mask_out_dir",
    help="Path where the tiled masks will be stored",
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

with tqdm(total=len(os.listdir(args.images_dir))) as pbar:
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(process_file, os.path.join(args.images_dir, filename))
            for filename in os.listdir(args.images_dir)
        ]
        # Wait for all futures to complete
        for _ in concurrent.futures.as_completed(futures):
            pbar.update(1)

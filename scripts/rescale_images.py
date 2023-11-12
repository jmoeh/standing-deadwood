import argparse
import os

import numpy as np
import rasterio


# add helper method to shorten cell size list
def shorten_list(arr, target):
    # Find the index of the first number greater than the target
    index = next((i for i, x in enumerate(arr) if x > target), None)

    if index is not None:
        # Shorten the list from the index onward
        shortened_list = arr[index:]
        return shortened_list
    else:
        # If no bigger number is found, return an empty array
        return []


# Add argument parser to get label dir, metadata file, and output dir
parser = argparse.ArgumentParser(description="Rescale images")
parser.add_argument(
    "images_dir",
    help="Path to the directory containing the to be rescaled images",
)
parser.add_argument(
    "-o",
    "--out_dir",
    help="Path where the rescaled images will be saved",
    default="./",
    required=False,
)
parser.add_argument(
    "-gw",
    "--gw-path",
    help="Path to the the gdal_warp binary",
    default="gdalwarp",
    required=False,
)
args = parser.parse_args()

# define desired cell width in meters
cell_widths = np.arange(0.04, 0.21, 0.02)

for filename in os.listdir(args.images_dir):
    filepath = os.path.join(args.images_dir, filename)
    if filename.endswith("8857.tif"):
        with rasterio.open(filepath) as image:
            # find desired cell widths for the given image
            image_cell_widths = shorten_list(cell_widths, image.transform[0])

            for image_cell_width in image_cell_widths:
                image_cell_width = round(image_cell_width, 2)
                os.system(
                    f"{args['gw-path']} -tr {image_cell_width} {image_cell_width} {filepath} {filepath[:-4]}_{'{0:.2f}'.format(image_cell_width)}.tif"
                )

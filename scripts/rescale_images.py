import argparse
import os
import concurrent.futures
import numpy as np
import rasterio
from tqdm import tqdm


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


def process_image(image_path, image_cell_width, gw_path):
    image_cell_width = round(image_cell_width, 2)
    scaled_image_path = image_path.replace(
        "_8857", "_8857_{0:.2f}".format(image_cell_width)
    )
    os.system(
        f"{gw_path} -q -tr {image_cell_width} {image_cell_width} {image_path} {scaled_image_path}"
    )


def process_file(filename, args, cell_widths):
    filepath = os.path.join(args.images_dir, filename)
    if filename.endswith("8857.tif") or filename.endswith("8857_mask.tif"):
        with rasterio.open(filepath) as image:
            # find desired cell widths for the given image
            image_cell_widths = shorten_list(cell_widths, image.transform[0])

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.jobs
            ) as executor:
                # Use ThreadPoolExecutor for parallel processing
                futures = [
                    executor.submit(process_image, filepath, cw, args.gw_path)
                    for cw in image_cell_widths
                ]
                # Wait for all futures to complete
                concurrent.futures.wait(futures)


# Add argument parser to get label dir, metadata file, and output dir
parser = argparse.ArgumentParser(description="Rescale images")
parser.add_argument(
    "images_dir",
    help="Path to the directory containing the to be rescaled images",
)
parser.add_argument(
    "-gw",
    "--gw_path",
    help="Path to the the gdal_warp binary",
    default="gdalwarp",
    required=False,
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

# define desired cell width in meters
cell_widths = np.arange(0.04, 0.21, 0.02)


# List all files in the images directory and process them in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
    futures = [
        executor.submit(process_file, filename, args, cell_widths)
        for filename in os.listdir(args.images_dir)
    ]
    # Wait for all futures to complete
    for _ in tqdm(
        concurrent.futures.as_completed(futures), total=len(os.listdir(args.images_dir))
    ):
        pass

import argparse
import concurrent.futures
import os

import tqdm


def process_file(in_file, out_dir):
    out_file = os.path.join(out_dir, in_file)
    if not os.path.exists(out_file):
        os.system(
            f"gdalwarp -q -co COMPRESS=DEFLATE -co TILED=YES -r bilinear -t_srs EPSG:8857 {os.path.join(args.images_dir,in_file)} {os.path.join(args.out_dir,in_file[:-4]+'_8857.tif')}"
        )


parser = argparse.ArgumentParser(description="Reproject images")
parser.add_argument(
    "images_dir",
    help="Path to the directory containing the to be rerojected images",
)
parser.add_argument(
    "-o",
    "--out_dir",
    help="Path to the the gdal_warp binary",
    default="./",
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
with tqdm.tqdm(total=len(os.listdir(args.images_dir))) as pbar:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(process_file, filename, args.out_dir)
            for filename in os.listdir(args.images_dir)
        ]
        # Wait for all futures to complete
        for _ in concurrent.futures.as_completed(futures):
            pbar.update(1)

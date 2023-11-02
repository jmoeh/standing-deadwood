import os
import pandas as pd
import rasterio
import argparse

# Add argument parser to get input dir and output file
parser = argparse.ArgumentParser(description="Generate orthophoto metadata.")
parser.add_argument(
    "input_dir", help="Path to the directory containing the GeoTIFF files"
)
parser.add_argument(
    "-o", "--output_file", help="Path to the output CSV file", default="output.csv"
)
args = parser.parse_args()

# Initialize an empty DataFrame
df = pd.DataFrame(
    columns=["filename", "left", "right", "bottom", "top", "width", "height"]
)

# Iterate over all GeoTIFF files in the directory
for filename in os.listdir(args.input_dir):
    if filename.endswith(".tif"):
        filepath = os.path.join(args.input_dir, filename)

        # Read the image
        with rasterio.open(filepath) as src:
            # Extract the bounds and resolution
            bounds = src.bounds
            meta = src.meta

        # Add a new row to the DataFrame
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "filename": filename,
                            "left": bounds.left,
                            "right": bounds.right,
                            "bottom": bounds.bottom,
                            "top": bounds.top,
                            "width": meta["width"],
                            "height": meta["height"],
                        }
                    ]
                ),
            ],
            axis=0,
            ignore_index=True,
        )

# Save the DataFrame to a CSV file
df.to_csv(args.output_file, index=False)

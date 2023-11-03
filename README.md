# Expanding UAV imagery for standing deadwood detection
## Getting started
* Python 3.8 is the preferred interpreter for this project otherwise `pytorch` won't be able to install
* Create a virtual environment by using `python3 -m venv venv` and activate it with `source venv/bin/activate`.
+ Install the specified dependencies with `pip install -r requirements.txt`.

## Scripts
* ### Generate metadata from ortophotos
    The exported csv file contains the geospatial bounds and pixel width and height of all the `*.tif` images in the given directory
    ```
    python scripts/generate_orthophoto_metadata.py -o [output_file] [images_dir]
    ```

* ### Generate masks for orthophoto labels
    The script will run through all `*.gpkg` files in the given label directory. To match the resolution of the given orthophotos the previously generated metadata csv needs to be specified.
    ```
    python scripts/generate_orthophoto_masks.py -m [metadata_file] [-o [out_dir]] [labels_dir]
    ```

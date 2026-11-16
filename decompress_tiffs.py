import argparse
import shutil
from pathlib import Path
from PIL import Image
from converter_paths import output_contains_input
# import numpy as np

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="read TIFF files and write them as uncompressed TIFF files")
parser.add_argument(
        "input_tiff_dir",
        help="Directory containing the TIFF files to be decompressed")
parser.add_argument(
        "output_tiff_dir",
        help="Directory where the decompressed TIFF files will be written")
parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory, if it already exists")

args = parser.parse_args()
idir = Path(args.input_tiff_dir)
odir = Path(args.output_tiff_dir)
overwrite = args.overwrite

if not idir.is_dir():
    parser.error("Input path must be an existing directory")
if output_contains_input(idir, odir):
    parser.error("Output directory must not contain the input directory")
files = sorted(idir.glob('*.tif'))
if not files:
    parser.error("Input directory contains no TIFF files")

if odir.exists():
    if overwrite:
        print("Removing existing",odir)
        shutil.rmtree(odir)
    else:
        print("Error: Directory",odir,"already exists")
        exit()

try:
    odir.mkdir()
except Exception as e:
    print("Error while creating",odir,":",e)
    exit()

# idir = r"G:\Vesuvius\Scroll1.volpkg\volumes\20230205180739"
# idir = r"F:\Vesuvius\Scroll1.volpkg\volumes_masked\20230205180739"
# odir = r"F:\Vesuvius\volumes_masked_uc"

for fl in files:
    oname = fl.name
    print("reading",oname,end='\r')
    # Can't use cv2 because OpenCV automatically compresses
    # TIFF files on writing
    iarr = Image.open(str(fl))
    ofl = odir / oname
    print("writing",oname,end='\r')
    iarr.save(str(ofl), compression=None)

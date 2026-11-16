from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scroll_to_ome import tifs2zarr


class CropTests(unittest.TestCase):
    def test_real_scroll_ranges(self):
        source = Path(__file__).parent / 'data/PHerc0800'
        original = np.stack([tifffile.imread(p) for p in sorted(source.glob('*.tif'))])
        cases = [
            (slice(3, 18), None, None),
            (None, slice(5, 23), None),
            (slice(3, 18), slice(5, 23), None),
            (None, None, slice(1, 3)),
            (None, None, None),
        ]
        for slices in cases:
            with self.subTest(slices=slices), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'crop.zarr'
                with redirect_stdout(io.StringIO()):
                    result = tifs2zarr(source, output, 2, slices=slices)
                self.assertIsNone(result)
                x, y, z = [s if s is not None else slice(None) for s in slices]
                np.testing.assert_array_equal(zarr.open(str(output), mode='r')[:], original[z, y, x])


if __name__ == '__main__':
    unittest.main()

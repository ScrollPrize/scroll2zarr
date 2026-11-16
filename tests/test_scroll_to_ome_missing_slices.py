from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scroll_to_ome import tifs2zarr


class MissingSliceTests(unittest.TestCase):
    def test_real_scroll_slices_keep_their_depth(self):
        fixture = Path(__file__).parent / 'data/PHerc0800'
        for indexes in ((0, 2), (0, 1, 2), (1, 2)):
            with self.subTest(indexes=indexes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'tiffs'
                source.mkdir()
                expected = np.zeros((3, 32, 32), dtype=np.uint8)
                for index in indexes:
                    image = fixture / f'{index:02}.tif'
                    shutil.copyfile(image, source / image.name)
                    expected[index] = tifffile.imread(image)
                with redirect_stdout(io.StringIO()):
                    result = tifs2zarr(source, root / 'out.zarr', 1)
                self.assertIsNone(result)
                np.testing.assert_array_equal(zarr.open(str(root / 'out.zarr'), mode='r')[:], expected)


if __name__ == '__main__':
    unittest.main()

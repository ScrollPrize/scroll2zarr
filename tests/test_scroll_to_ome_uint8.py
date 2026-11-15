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


class ByteConversionTests(unittest.TestCase):
    def test_real_uint16_crop_is_divided_once(self):
        fixture = Path(__file__).parent / 'data/PHercParis4/00000.tif'
        image = tifffile.imread(fixture)
        for planes in (1, 3, 4):
            for obytes in (0, 1):
                with self.subTest(planes=planes, obytes=obytes), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / 'tiffs'
                    source.mkdir()
                    for index in range(planes):
                        shutil.copyfile(fixture, source / f'{index:05}.tif')
                    with redirect_stdout(io.StringIO()):
                        result = tifs2zarr(source, root / 'out.zarr', 2, obytes=obytes)
                    self.assertIsNone(result)
                    expected = np.stack([image] * planes)
                    if obytes == 1:
                        expected = (expected // 256).astype(np.uint8)
                    actual = zarr.open(str(root / 'out.zarr'), mode='r')
                    self.assertEqual(actual.dtype, expected.dtype)
                    np.testing.assert_array_equal(actual[:], expected)


if __name__ == '__main__':
    unittest.main()

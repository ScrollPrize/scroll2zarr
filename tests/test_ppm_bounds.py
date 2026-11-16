from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppm_to_layers import process_block, trilinear


class PpmBoundsTests(unittest.TestCase):
    def setUp(self):
        fixture = Path(__file__).parent / 'data/PHerc0800'
        self.image = np.stack([tifffile.imread(p) for p in sorted(fixture.glob('*.tif'))]).astype(np.float32)

    def reference(self, points, image):
        interpolate = RegularGridInterpolator(tuple(np.arange(size) for size in image.shape), image, bounds_error=False, fill_value=0)
        finite = np.isfinite(points).all(axis=-1)
        output = np.zeros(points.shape[:-1])
        output[finite] = interpolate(points[finite])
        return output

    def test_interpolation_at_and_outside_real_ct_faces(self):
        points = [[1, 4, 4], [1.25, 4.5, 4.25], [0, 0, 0]]
        points.extend([z, y, x] for z in (0, 1, 2) for y in (0, 15, 31) for x in (0, 15, 31))
        for axis, size in enumerate(self.image.shape):
            for value in (-100, -.25, size - .75, size + 100):
                point = [1, 15, 15]
                point[axis] = value
                points.append(point)
        points.extend([[np.nan, 15, 15], [1, np.inf, 15], [1, 15, -np.inf]])
        for index, point in enumerate(points):
            with self.subTest(point=index):
                coordinates = np.array([point], dtype=np.float64)
                np.testing.assert_allclose(trilinear(coordinates, self.image), self.reference(coordinates, self.image), rtol=0, atol=1e-5)

    def test_real_ct_blocks_keep_valid_pixels_at_volume_edges(self):
        cases = [
            ('interior', (4, 4, 1)),
            ('last-x', (30, 4, 1)),
            ('last-y', (4, 30, 1)),
            ('last-z', (4, 4, 2)),
            ('corner', (30, 30, 2)),
            ('lower-x', (-.5, 4, 1)),
            ('upper-x', (30.5, 4, 1)),
            ('below-x', (-4, 4, 1)),
            ('above-x', (40, 4, 1)),
            ('single-z', (4, 4, 0)),
            ('invalid-far', (4, 4, 1)),
            ('invalid-nan', (4, 4, 1)),
        ]
        for name, origin in cases:
            for disk in (False, True):
                with self.subTest(case=name, disk=disk), tempfile.TemporaryDirectory() as directory:
                    image = self.image[1:2] if name == 'single-z' else self.image
                    xyz = np.zeros((2, 2, 3), dtype=np.float64)
                    y, x = np.mgrid[:2, :2]
                    xyz[..., 0] = origin[0] + x
                    xyz[..., 1] = origin[1] + y
                    xyz[..., 2] = origin[2]
                    normals = np.zeros_like(xyz)
                    normals[..., 2] = 1
                    valid = np.ones((2, 2), dtype=bool)
                    if name.startswith('invalid-'):
                        xyz[0, 0] = np.nan if name == 'invalid-nan' else 1e6
                        normals[0, 0] = 0
                        valid[0, 0] = False
                    if disk:
                        data = zarr.open(str(Path(directory) / 'scan.zarr'), mode='w', shape=image.shape, chunks=(1, 8, 8), dtype=image.dtype)
                        data[:] = image
                    else:
                        data = image
                    layers = np.zeros((1, 2, 2), dtype=np.uint16)
                    stack = np.zeros((2, 2), dtype=np.uint16)
                    with redirect_stdout(io.StringIO()):
                        process_block(np.array([origin[2], 0, 0]), 2, (xyz, normals, valid), data, (np.array([0]), slice(0, 1)), (layers, stack), 0)
                    expected = self.reference(xyz[..., ::-1], image)
                    expected[~valid] = 0
                    np.testing.assert_array_equal(layers[0], expected.astype(np.uint16))
                    np.testing.assert_array_equal(stack, expected.astype(np.uint16))


if __name__ == '__main__':
    unittest.main()

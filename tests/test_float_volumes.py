from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import skimage.transform
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zarr_to_ome


class FloatVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parent / 'data/PHerc0800'
        cls.ct = np.stack([tifffile.imread(p) for p in sorted(fixture.glob('*.tif'))])

    def test_copy_preserves_float_samples(self):
        for dtype in (np.float32, np.float64):
            image = self.ct.astype(dtype) / 255
            for channel in (None, 0, 1):
                with self.subTest(dtype=dtype, channel=channel), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    data = image if channel is None else np.stack((image, image - 1))
                    source = zarr.open(str(path / 'input.zarr'), mode='w', shape=data.shape, chunks=data.shape, dtype=dtype)
                    source[:] = data
                    with redirect_stdout(io.StringIO()):
                        error = zarr_to_ome.zarr2zarr(str(path / 'input.zarr'), str(path / 'output.zarr'), channel=channel or 0)
                    self.assertIsNone(error)
                    actual = zarr.open(str(path / 'output.zarr'), mode='r')
                    self.assertEqual(actual.dtype, dtype)
                    np.testing.assert_array_equal(actual[:], image if channel is None else data[channel])

    def test_float_pyramids_keep_fractional_values(self):
        for dtype in (np.float32, np.float64):
            image = self.ct.astype(dtype) / 255
            padded = np.pad(image, ((0, 1), (0, 0), (0, 0)), mode='symmetric')
            blocks = padded.reshape(2, 2, 16, 2, 16, 2)
            expected = {
                'nearest': padded[::2, ::2, ::2],
                'mean': blocks.mean(axis=(1, 3, 5)),
                'max': blocks.max(axis=(1, 3, 5)),
                'gaussian': skimage.transform.rescale(padded, .5, preserve_range=True),
            }
            for algorithm, values in expected.items():
                with self.subTest(dtype=dtype, algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    source = zarr.open(str(path / '0'), mode='w', shape=image.shape, chunks=image.shape, dtype=dtype)
                    source[:] = image
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        zarr_to_ome.resize(path, 0, 2, algorithm)
                    actual = zarr.open(str(path / '1'), mode='r')
                    self.assertEqual(actual.dtype, dtype)
                    np.testing.assert_allclose(actual[:], values.astype(dtype), rtol=1e-7, atol=0)

    def test_negative_voxels_are_not_an_empty_block(self):
        image = self.ct.astype(np.float32)
        image = (image - image.max()) / 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = zarr.open(str(path / '0'), mode='w', shape=image.shape, chunks=image.shape, dtype=image.dtype)
            source[:] = image
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                zarr_to_ome.resize(path, 0, 2, 'nearest')
            np.testing.assert_array_equal(zarr.open(str(path / '1'), mode='r')[:], image[::2, ::2, ::2])

    def test_nonfloating_pyramids_keep_rounding(self):
        for image in (self.ct, self.ct.astype(np.uint16) * 257, self.ct > np.median(self.ct)):
            padded = np.pad(image, ((0, 1), (0, 0), (0, 0)), mode='symmetric')
            blocks = padded.reshape(2, 2, 16, 2, 16, 2)
            expected = {
                'nearest': padded[::2, ::2, ::2],
                'mean': blocks.mean(axis=(1, 3, 5)),
                'max': blocks.max(axis=(1, 3, 5)),
                'gaussian': skimage.transform.rescale(padded, .5, preserve_range=True),
            }
            for algorithm, values in expected.items():
                with self.subTest(dtype=image.dtype, algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    source = zarr.open(str(path / '0'), mode='w', shape=image.shape, chunks=image.shape, dtype=image.dtype)
                    source[:] = image
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        zarr_to_ome.resize(path, 0, 2, algorithm)
                    np.testing.assert_array_equal(zarr.open(str(path / '1'), mode='r')[:], np.round(values).astype(image.dtype))

    def test_uint16_conversion_keeps_integer_rounding(self):
        image = self.ct.astype(np.uint16) * 257
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = zarr.open(str(path / 'input.zarr'), mode='w', shape=image.shape, chunks=image.shape, dtype=image.dtype)
            source[:] = image
            with redirect_stdout(io.StringIO()):
                error = zarr_to_ome.zarr2zarr(str(path / 'input.zarr'), str(path / 'output.zarr'), dtype=np.uint8)
            self.assertIsNone(error)
            np.testing.assert_array_equal(zarr.open(str(path / 'output.zarr'), mode='r')[:], self.ct)
            padded = np.pad(self.ct, ((0, 1), (0, 0), (0, 0)), mode='symmetric')
            expected = np.round(padded.reshape(2, 2, 16, 2, 16, 2).mean(axis=(1, 3, 5))).astype(np.uint8)
            level0 = zarr.open(str(path / '0'), mode='w', shape=self.ct.shape, chunks=self.ct.shape, dtype=np.uint8)
            level0[:] = self.ct
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                zarr_to_ome.resize(path, 0, 2, 'mean')
            np.testing.assert_array_equal(zarr.open(str(path / '1'), mode='r')[:], expected)


if __name__ == '__main__':
    unittest.main()

from contextlib import redirect_stdout, redirect_stderr
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
import scroll_to_ome
import zarr_to_ome


class DownsampleTests(unittest.TestCase):
    def test_real_ct_matches_whole_volume_filter(self):
        fixture = Path(__file__).parent / 'data/PHerc0800'
        image = np.stack([tifffile.imread(p) for p in sorted(fixture.glob('*.tif'))])[:, :31, :29]
        padded = np.pad(image, ((0, 1), (0, 1), (0, 1)), mode='symmetric')
        expected = {
            'gaussian': np.round(skimage.transform.rescale(padded, .5, preserve_range=True)),
            'mean': np.round(skimage.transform.downscale_local_mean(padded, (2, 2, 2))),
            'nearest': padded[::2, ::2, ::2],
            'max': padded.reshape((2, 2, 16, 2, 15, 2)).max(axis=(1, 3, 5)),
        }
        for module in (scroll_to_ome, zarr_to_ome):
            for algorithm in ('gaussian', 'mean', 'nearest', 'max'):
                if module is scroll_to_ome and algorithm == 'max':
                    continue
                for size in (4, 8, 32):
                    with self.subTest(module=module.__name__, algorithm=algorithm, size=size), tempfile.TemporaryDirectory() as directory:
                        path = Path(directory)
                        data = zarr.open(str(path / '0'), mode='w', shape=image.shape, chunks=(size,) * 3, dtype=image.dtype)
                        data[:] = image
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            module.resize(path, 0, 2, algorithm)
                        actual = zarr.open(str(path / '1'), mode='r')[:]
                        np.testing.assert_array_equal(actual, expected[algorithm].astype(image.dtype))


if __name__ == '__main__':
    unittest.main()

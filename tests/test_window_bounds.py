from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numcodecs import Blosc
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zarr_to_ome import zarr2zarr


class WindowBoundsTests(unittest.TestCase):
    def test_real_ct_windows_copy_only_source_coordinates(self):
        fixture = Path(__file__).parent / 'data/PHerc0800'
        image = np.stack([tifffile.imread(path) for path in sorted(fixture.glob('*.tif'))])
        cases = [
            ('upper-x', (0, 40), (0, 0, 0)),
            ('upper-all', (0, 40), (0, 0, 0)),
            ('upper-border', (31, 40), (0, 0, 0)),
            ('fully-outside', (40, 44), (0, 0, 0)),
            ('upper-x-negative-shift', (20, 40), (-24, 0, 0)),
            ('inside-shifted', (0, 20), (8, 0, 0)),
            ('identity', None, (0, 0, 0)),
            ('inside-negative-shift', (4, 20), (-4, 0, 0)),
        ]
        for compressed in (False, True):
            for ome in (False, True):
                for chunk in (4, 16):
                    for name, bounds, shift_xyz in cases:
                        with self.subTest(case=name, compressed=compressed, ome=ome, chunk=chunk), tempfile.TemporaryDirectory() as directory:
                            slices_xyz = [slice(*bounds) if bounds else None, None, None]
                            if name == 'upper-all':
                                slices_xyz[1:] = [slice(0, 35), slice(0, 5)]
                            shift = shift_xyz[::-1]
                            slices = slices_xyz[::-1]
                            starts = [part.start if part and part.start is not None else 0 for part in slices]
                            stops = [part.stop if part and part.stop is not None else size for part, size in zip(slices, image.shape)]
                            shape = tuple(max(0, stop + offset) for stop, offset in zip(stops, shift))
                            coords = [np.arange(size) - offset for size, offset in zip(shape, shift)]
                            valid = [(coord >= 0) & (coord < size) & (coord >= start) & (coord < stop) for coord, size, start, stop in zip(coords, image.shape, starts, stops)]
                            mask = valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :]
                            expected = np.where(mask, image[np.ix_(*(np.clip(coord, 0, size - 1) for coord, size in zip(coords, image.shape)))], 0)
                            path = Path(directory)
                            source = path / 'source.zarr'
                            if ome:
                                zarr.group(str(source))
                            array = zarr.open(str(source / '0' if ome else source), mode='w', shape=image.shape, chunks=(2, 8, 8), dtype=image.dtype, compressor=Blosc() if compressed else None)
                            array[:] = image
                            with redirect_stdout(io.StringIO()):
                                error = zarr2zarr(str(source), path / 'output.zarr', shift=shift_xyz, slices=slices_xyz, chunk_size=chunk)
                            self.assertIsNone(error)
                            actual = zarr.open(str(path / 'output.zarr'), mode='r')[:]
                            np.testing.assert_array_equal(actual, expected)


    def test_multichannel_ct_window_uses_the_selected_source_extent(self):
        fixture = Path(__file__).parent / 'data/PHerc0800'
        image = np.stack([tifffile.imread(path) for path in sorted(fixture.glob('*.tif'))])
        channels = np.stack((image[:-1], image[1:]))
        expected = np.pad(channels[1], ((0, 0), (0, 0), (0, 8)))
        for compressed in (False, True):
            for ome in (False, True):
                for chunk in (4, 16):
                    with self.subTest(compressed=compressed, ome=ome, chunk=chunk), tempfile.TemporaryDirectory() as directory:
                        path = Path(directory)
                        source = path / 'source.zarr'
                        if ome:
                            zarr.group(str(source))
                        array = zarr.open(str(source / '0' if ome else source), mode='w', shape=channels.shape, chunks=(1, 2, 8, 8), dtype=image.dtype, compressor=Blosc() if compressed else None)
                        array[:] = channels
                        with redirect_stdout(io.StringIO()):
                            error = zarr2zarr(str(source), path / 'output.zarr', slices=(slice(0, 40), None, None), chunk_size=chunk, channel=1)
                        self.assertIsNone(error)
                        actual = zarr.open(str(path / 'output.zarr'), mode='r')[:]
                        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()

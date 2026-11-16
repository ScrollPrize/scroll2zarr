from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tifffile
import zarr


class PpmWorkerTests(unittest.TestCase):
    def prepare(self, root, group, corrupt):
        fixture = Path(__file__).parent / 'data/PHerc0800'
        image = np.stack([tifffile.imread(p) for p in sorted(fixture.glob('*.tif'))])
        store = root / 'scan.zarr'
        if group:
            array = zarr.open_group(str(store), mode='w').create_dataset('0', shape=image.shape, chunks=(3, 16, 16), dtype=image.dtype)
            chunks = store / '0'
        else:
            array = zarr.open(str(store), mode='w', shape=image.shape, chunks=(3, 16, 16), dtype=image.dtype)
            chunks = store
        array[:] = image
        if corrupt:
            chunk = chunks / '0.0.0'
            encoded = chunk.read_bytes()
            chunk.write_bytes(encoded[:8])
        surface = np.zeros((4, 4, 6), dtype='<f8')
        y, x = np.mgrid[4:8, 4:8]
        surface[..., 0] = x
        surface[..., 1] = y
        surface[..., 2] = 1
        surface[..., 5] = 1
        ppm = root / 'plane.ppm'
        ppm.write_bytes(b'width: 4\nheight: 4\ndim: 6\nordered: true\ntype: double\nversion: 1\n<>\n' + surface.tobytes())
        return ppm, store, image[1, 4:8, 4:8].astype(np.uint16)

    def convert(self, root, ppm, store, block, stack):
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'ppm_to_layers.py'), str(ppm), str(store), str(root / 'layers')]
        if stack:
            command.append(str(root / 'stack.tif'))
        command.extend(['--number_of_layers', '1', '--number_of_stacked_layers', '1', '--block_size', str(block), '--zarr_cache_max_size_gb', '1'])
        return subprocess.run(command, capture_output=True, text=True, timeout=30)

    def test_real_ct_layers_are_unchanged(self):
        for group in (False, True):
            for block in (1, 2, 4):
                for stack in (False, True):
                    with self.subTest(group=group, block=block, stack=stack), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        ppm, store, expected = self.prepare(root, group, False)
                        result = self.convert(root, ppm, store, block, stack)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        np.testing.assert_array_equal(tifffile.imread(root / 'layers/00.tif'), expected)
                        if stack:
                            np.testing.assert_array_equal(tifffile.imread(root / 'stack.tif'), expected)

    def test_corrupt_real_ct_chunk_fails_before_writing_layers(self):
        for group in (False, True):
            for block in (1, 2, 4):
                for stack in (False, True):
                    with self.subTest(group=group, block=block, stack=stack), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        ppm, store, _ = self.prepare(root, group, True)
                        result = self.convert(root, ppm, store, block, stack)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('decompress', result.stderr.lower())
                        self.assertEqual(list((root / 'layers').glob('*.tif')), [])
                        self.assertFalse((root / 'stack.tif').exists())


if __name__ == '__main__':
    unittest.main()

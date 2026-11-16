import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile


class DecompressInputTests(unittest.TestCase):
    def populate(self, path):
        path.mkdir(parents=True)
        fixture = Path(__file__).parent / 'data/PHerc0800'
        for image in fixture.glob('*.tif'):
            with Image.open(image) as source:
                source.save(path / image.name, compression='tiff_lzw')

    def snapshot(self, root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file()}

    def run_converter(self, root, source, destination, overwrite):
        for target in (source.resolve(), destination.resolve()):
            self.assertIn(root.resolve(), target.parents)
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'decompress_tiffs.py'), str(source), str(destination)]
        if overwrite:
            command.append('--overwrite')
        return subprocess.run(command, capture_output=True, text=True, timeout=30)

    def test_invalid_input_preserves_source_and_existing_output(self):
        for case in ('same', 'same-alias', 'parent', 'missing', 'empty'):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix='vesuvius-tiff-input-') as directory:
                root = Path(directory).resolve()
                source = root / 'tree/source'
                destination = root / 'output'
                self.populate(source)
                if case == 'same':
                    destination = source
                elif case == 'same-alias':
                    destination = source / '../source'
                elif case == 'parent':
                    destination = source.parent
                else:
                    destination.mkdir()
                (destination / 'keep.txt').write_bytes(b'previous output')
                if case == 'missing':
                    source = root / 'missing'
                elif case == 'empty':
                    source = root / 'empty'
                    source.mkdir()
                before = self.snapshot(root)
                result = self.run_converter(root, source, destination, True)
                self.assertEqual(self.snapshot(root), before)
                self.assertNotEqual(result.returncode, 0)

    def test_real_ct_converts_to_safe_output_directories(self):
        for case in ('new', 'replace', 'child'):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix='vesuvius-tiff-output-') as directory:
                root = Path(directory).resolve()
                source = root / 'source'
                self.populate(source)
                before = self.snapshot(source)
                destination = source / 'uncompressed' if case == 'child' else root / 'output'
                if case == 'replace':
                    destination.mkdir()
                    (destination / 'old.txt').write_bytes(b'old output')
                result = self.run_converter(root, source, destination, case == 'replace')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(list(destination.glob('*.tif'))), 3)
                self.assertFalse((destination / 'old.txt').exists())
                for name, digest in before.items():
                    self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest)
                    with Image.open(source / name) as image:
                        self.assertEqual(image.tag_v2.get(259), 5)
                    np.testing.assert_array_equal(tifffile.imread(source / name), tifffile.imread(destination / name))
                    with Image.open(destination / name) as image:
                        self.assertEqual(image.tag_v2.get(259), 1)


if __name__ == '__main__':
    unittest.main()

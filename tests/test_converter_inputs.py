from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Thread
import unittest

import numpy as np
import tifffile
import zarr


class ConverterInputTests(unittest.TestCase):
    def image(self):
        fixture = Path(__file__).parent / 'data/PHerc0800'
        return np.stack([tifffile.imread(fixture / f'{index:02}.tif') for index in range(2)])

    def populate(self, path, tool):
        image = self.image()
        if tool == 'tiff':
            path.mkdir(parents=True)
            fixture = Path(__file__).parent / 'data/PHerc0800'
            for index in range(2):
                shutil.copyfile(fixture / f'{index:02}.tif', path / f'{index:02}.tif')
        else:
            zarr.open(str(path), mode='w', shape=image.shape, chunks=(2, 16, 16), dtype=image.dtype)[:] = image

    def snapshot(self, root):
        return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*') if path.is_file()}

    def run_converter(self, root, tool, source, output, *options, env=None):
        self.assertIn(root.resolve(), output.resolve().parents)
        script = 'scroll_to_ome.py' if tool == 'tiff' else 'zarr_to_ome.py'
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / script), str(source), str(output), '--num_threads', '1', '--algorithm', 'nearest', *options]
        return subprocess.run(command, capture_output=True, text=True, timeout=30, env=env)

    def invalid_case(self, root, tool, case):
        source = root / 'parent.zarr/source.zarr' if case == 'parent' else root / 'source.zarr'
        self.populate(source, tool)
        output = source.parent if case == 'parent' else source
        argument = str(source)
        if case == 'same-alias':
            output = source / '..' / source.name
        elif case == 'file-url':
            argument = source.as_uri()
        elif case in ('missing', 'file-input'):
            output = root / 'output.zarr'
            output.mkdir()
            argument = str(root / 'missing.zarr') if case == 'missing' else str(source / '00.tif')
        (output / 'keep.txt').write_bytes(b'previous output')
        return argument, output

    def test_invalid_paths_preserve_all_existing_files(self):
        cases = [('tiff', name) for name in ('same', 'same-alias', 'parent', 'file-input', 'zarr-only-rebuild')]
        cases += [('zarr', name) for name in ('same', 'same-alias', 'parent', 'file-url', 'missing')]
        for tool, case in cases:
            with self.subTest(tool=tool, case=case), tempfile.TemporaryDirectory(prefix='vesuvius-converter-input-') as directory:
                root = Path(directory).resolve()
                source, output = self.invalid_case(root, tool, case)
                before = self.snapshot(root)
                options = ['--overwrite', '--nlevels', '1']
                if case == 'zarr-only-rebuild':
                    options.extend(['--zarr_only', '--first_new_level', '1'])
                result = self.run_converter(root, tool, source, output, *options)
                self.assertEqual(self.snapshot(root), before)
                self.assertNotEqual(result.returncode, 0)

    def test_real_ct_conversions_preserve_source_pixels(self):
        for tool in ('tiff', 'zarr'):
            for location in ('new', 'replace', 'child'):
                for levels in (1, 2):
                    for uri in ((False, True) if tool == 'zarr' else (False,)):
                        with self.subTest(tool=tool, location=location, levels=levels, uri=uri), tempfile.TemporaryDirectory(prefix='vesuvius-converter-output-') as directory:
                            root = Path(directory).resolve()
                            source = root / 'source.zarr'
                            self.populate(source, tool)
                            before = self.snapshot(source)
                            output = source / 'output.zarr' if location == 'child' else root / 'output.zarr'
                            if location == 'replace':
                                output.mkdir()
                                (output / 'old.txt').write_bytes(b'previous output')
                            options = ['--nlevels', str(levels)]
                            if location == 'replace':
                                options.append('--overwrite')
                            if tool == 'tiff' and levels == 1:
                                options.append('--zarr_only')
                            result = self.run_converter(root, tool, source.as_uri() if uri else source, output, *options)
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            level0 = output if levels == 1 else output / '0'
                            np.testing.assert_array_equal(zarr.open(str(level0), mode='r')[:], self.image())
                            if levels == 2:
                                np.testing.assert_array_equal(zarr.open(str(output / '1'), mode='r')[:], self.image()[::2, ::2, ::2])
                            after = self.snapshot(source)
                            self.assertTrue(all(after.get(name) == digest for name, digest in before.items()))
                            self.assertFalse((output / 'old.txt').exists())

    def test_rebuild_does_not_treat_output_as_input(self):
        for tool in ('tiff', 'zarr'):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory(prefix='vesuvius-converter-rebuild-') as directory:
                root = Path(directory).resolve()
                output = root / 'output.zarr'
                zarr.open_group(str(output), mode='w')
                self.populate(output / '0', 'zarr')
                before = self.snapshot(output / '0')
                options = ['--first_new_level', '1'] if tool == 'tiff' else ['--rebuild']
                result = self.run_converter(root, tool, output, output, '--nlevels', '2', *options)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.snapshot(output / '0'), before)
                np.testing.assert_array_equal(zarr.open(str(output / '1'), mode='r')[:], self.image()[::2, ::2, ::2])

    def test_http_input_still_converts(self):
        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

        with tempfile.TemporaryDirectory(prefix='vesuvius-converter-http-') as directory:
            root = Path(directory).resolve()
            self.populate(root / 'source.zarr', 'zarr')
            handler = partial(Handler, directory=str(root))
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                temporary = root / 'tmp'
                temporary.mkdir()
                env = os.environ.copy()
                env.update({'TEMP': str(temporary), 'TMP': str(temporary), 'TMPDIR': str(temporary)})
                for transport, prefix in [('http', ''), ('filecache', 'filecache::')]:
                    with self.subTest(transport=transport):
                        output = root / f'{transport}.zarr'
                        result = self.run_converter(root, 'zarr', f'{prefix}http://127.0.0.1:{server.server_port}/source.zarr', output, '--nlevels', '1', env=env)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        np.testing.assert_array_equal(zarr.open(str(output), mode='r')[:], self.image())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()

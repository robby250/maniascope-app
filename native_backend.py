"""Build and load optional extensions from the same Python calculator sources.

Nothing is compiled at app startup. A missing/stale build uses Python. Artifacts
are isolated by source and Python ABI; a running process keeps its loaded ID.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import recdata

ROOT = Path(__file__).resolve().parent
MODULES = recdata.CALCULATOR_MODULES
# recdata.py is not compiled: listing it made every playlist/stats edit discard the build (G835LX ran
# 2.5× slower Python from 2026-09-28). activate() still checks the calculator identity it computes.
INPUTS = tuple(name+'.py' for name in MODULES) + (
    'skill_calc.pxd', 'gestures.pxd', 'execution.pxd', 'calib/difficulty.json', 'native_backend.py')
ABI = sysconfig.get_config_var('SOABI')
SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')
FILES = tuple(name+SUFFIX for name in MODULES) + ('calib/difficulty.json',)
CALCULATOR = None


def hashes(root, names):
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def location(inputs, directory=None):
    import paths
    base = Path(directory) if directory is not None else Path(paths.CACHE)/'native'
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return base/ABI/identity


def activate(directory=None):
    """Load only a complete matching build, before any calculator import."""
    global CALCULATOR
    if CALCULATOR is not None:
        return True
    if any(name in sys.modules for name in MODULES):
        return False
    try:
        inputs = hashes(ROOT, INPUTS)
        path = location(inputs, directory)
        manifest = json.loads((path/'manifest.json').read_text())
        calculator = manifest['calculator']
        expected = recdata._calculator_identity(
            [ROOT/(name+'.py') for name in recdata.CALCULATOR_MODULES] + [ROOT/'calib/difficulty.json'])
        if (manifest['abi'] != ABI or manifest['inputs'] != inputs
                or manifest['artifacts'] != hashes(path, FILES)
                or calculator != expected):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    sys.path.insert(0, str(path))
    CALCULATOR = calculator
    return True


def build(directory=None):
    """Requires Cython/setuptools only in the build interpreter's environment."""
    import shutil
    import tempfile
    from setuptools import setup
    from Cython.Build import cythonize
    import Cython
    inputs = hashes(ROOT, INPUTS)
    destination = location(inputs, directory)
    if destination.exists():
        manifest = json.loads((destination/'manifest.json').read_text())
        if (manifest['inputs'] != inputs or manifest['abi'] != ABI
                or manifest['artifacts'] != hashes(destination, FILES)
                or manifest['calculator'] != recdata.calc_id()):
            raise ValueError('Existing native build is incomplete or changed: '+str(destination))
        return destination
    calculator = recdata.calc_id()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.build-', dir=destination.parent) as temp:
        work = Path(temp)/'source'
        output = Path(temp)/'output'
        for name in INPUTS:
            target = work/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        previous = Path.cwd()
        try:
            os.chdir(work)
            extensions = cythonize([name+'.py' for name in MODULES],
                                  compiler_directives={'language_level': 3,
                                      'annotation_typing': True, 'infer_types': True})
            for extension in extensions:
                extension.extra_compile_args = ['-g0']
            setup(name='maniascope-native', ext_modules=extensions,
                  script_args=['build_ext', '--build-lib', str(output),
                               '--build-temp', str(Path(temp)/'objects')])
        finally:
            os.chdir(previous)
        if hashes(ROOT, INPUTS) != inputs or hashes(work, INPUTS) != inputs:
            raise RuntimeError('Calculator source changed during the build')
        (output/'calib').mkdir(exist_ok=True)
        shutil.copy2(work/'calib/difficulty.json', output/'calib/difficulty.json')
        manifest = dict(abi=ABI, inputs=inputs, artifacts=hashes(output, FILES),
                        calculator=calculator, python=sys.version, cython=Cython.__version__)
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        os.rename(output, destination)
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, help='Build cache root (default: ~/.cache/maniascope/native)')
    args = parser.parse_args()
    print(build(args.directory))

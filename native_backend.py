"""Build and load the Rust calculator (native/, module maniascope_native).

A build is keyed by the hash of the crate sources, so a git pull or a Dropbox sync never loads a stale
calculator. Packaged apps import the module they bundle. Source installs use
<cache>/native/rust/<source hash>/: built here when cargo is present (G533QR), copied in otherwise
(G835LX gets it by rsync; Linux friends from the release asset, see setup.sh).
"""
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WINDOWS = sys.platform == 'win32'
ARTIFACT = 'maniascope_native.pyd' if WINDOWS else 'maniascope_native.abi3.so'


def sources(root=None):
    crate = Path(root or ROOT)/'native'
    return [crate/'Cargo.toml', crate/'Cargo.lock'] + sorted((crate/'src').glob('*.rs'))


def identity_files(root=None):
    """What calc_id hashes: crate sources, Python faces, fitted stages."""
    import recdata
    root = Path(root or ROOT)
    # The bulk predictor and the thread pool share the crate, not the identity: neither changes a calculator
    # result, and a change to them must not re-analyse every library.
    return ([p for p in sources(root) if p.name not in ('predict.rs', 'background.rs')]
            + [root/(name+'.py') for name in recdata.CALCULATOR_MODULES] + [root/'calib/difficulty.json'])


def source_hash():
    h = hashlib.sha256()
    for path in sources():
        h.update(path.relative_to(ROOT/'native').as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return h.hexdigest()[:16]


def _cargo():
    """rustup's cargo before a distro one (Mint's apt cargo is too old for the lock file)."""
    rustup = Path.home()/'.cargo'/'bin'/('cargo.exe' if WINDOWS else 'cargo')
    return str(rustup) if rustup.is_file() else shutil.which('cargo')


def location(directory=None):
    import paths
    base = Path(directory) if directory is not None else Path(paths.CACHE)/'native'/'rust'
    return base/source_hash()


def load(directory=None):
    """The calculator module; builds it first when it is missing and cargo is available."""
    if 'maniascope_native' in sys.modules:
        return sys.modules['maniascope_native']
    if getattr(sys, 'frozen', False):
        return importlib.import_module('maniascope_native')
    path = location(directory)
    if not (path/ARTIFACT).is_file():
        if not _cargo():
            raise ImportError(f'The ManiaScope calculator is not built ({path/ARTIFACT} missing). '
                              'Run ./setup.sh, or install Rust (rustup.rs) and run python3 native_backend.py.')
        print('Building the ManiaScope calculator (one-time, ~2 min)…', file=sys.stderr, flush=True)
        build(directory)
    sys.path.insert(0, str(path))
    try:
        return importlib.import_module('maniascope_native')
    finally:
        sys.path.remove(str(path))


def build(directory=None):
    """cargo build --release into the source-hash directory (atomic rename; a concurrent build wins)."""
    destination = location(directory)
    if (destination/ARTIFACT).is_file():
        return destination
    import paths
    before = source_hash()
    cargo = _cargo()
    target = Path(paths.CACHE)/'native'/'target'
    subprocess.run([cargo, 'build', '--release', '--manifest-path', str(ROOT/'native'/'Cargo.toml')],
                   check=True, env=dict(os.environ, CARGO_TARGET_DIR=str(target), PYO3_PYTHON=sys.executable))
    if source_hash() != before:
        raise RuntimeError('Calculator source changed during the build')
    built = target/'release'/('maniascope_native.dll' if WINDOWS else 'libmaniascope_native.so')
    temp = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    temp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, temp/ARTIFACT)
    try:
        os.rename(temp, destination)
    except OSError:
        shutil.rmtree(temp)
        if not (destination/ARTIFACT).is_file():
            raise
    return destination


def calc_id():
    """Identity of the calculator: crate sources, Python faces, fitted stages and the skill list."""
    if getattr(sys, 'frozen', False):
        # Packaged modules have no source files to hash; the build records the identity it froze.
        with open(os.path.join(sys._MEIPASS, 'calc_id.txt'), encoding='utf-8') as fh:
            return fh.read().strip()
    import recdata
    return recdata._calculator_identity(identity_files())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, help='Build root (default: <cache>/native/rust)')
    args = parser.parse_args()
    print(build(args.directory))

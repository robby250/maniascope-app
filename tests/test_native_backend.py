"""A build is keyed by its crate sources; a missing build never loads a stale one."""
import pytest

import native_backend as N


def crate(root, text):
    (root/'native/src').mkdir(parents=True, exist_ok=True)
    (root/'native/Cargo.toml').write_text('[package]\n')
    (root/'native/Cargo.lock').write_text('')
    (root/'native/src/lib.rs').write_text(text)


def test_build_location_follows_every_crate_source(tmp_path, monkeypatch):
    crate(tmp_path, 'fn a() {}\n')
    monkeypatch.setattr(N, 'ROOT', tmp_path)
    first = N.location(tmp_path/'cache')
    assert N.location(tmp_path/'cache') == first
    (tmp_path/'native/src/lib.rs').write_text('fn b() {}\n')
    assert N.location(tmp_path/'cache') != first
    (tmp_path/'native/src/extra.rs').write_text('')
    (tmp_path/'native/src/lib.rs').write_text('fn a() {}\n')
    assert N.location(tmp_path/'cache') != first


def test_missing_build_without_cargo_is_an_import_error(tmp_path, monkeypatch):
    crate(tmp_path, 'fn a() {}\n')
    monkeypatch.setattr(N, 'ROOT', tmp_path)
    monkeypatch.setattr(N, '_cargo', lambda: None)
    monkeypatch.delitem(__import__('sys').modules, 'maniascope_native', raising=False)
    with pytest.raises(ImportError, match='not built'):
        N.load(tmp_path/'cache')

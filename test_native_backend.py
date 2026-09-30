"""The optional build must never mix sources, ABIs or calculator identities."""
import json
import sys

import native_backend as N
import recdata


def test_native_loader_checks_complete_build_and_keeps_running_identity(tmp_path, monkeypatch):
    root = tmp_path/'source'
    root.mkdir()
    (root/'probe.py').write_text('value = 1\n')
    (root/'calib').mkdir()
    (root/'calib/difficulty.json').write_text('{}')
    monkeypatch.setattr(N, 'ROOT', root)
    monkeypatch.setattr(N, 'MODULES', ('probe',))
    monkeypatch.setattr(N, 'INPUTS', ('probe.py', 'calib/difficulty.json'))
    monkeypatch.setattr(N, 'FILES', ('probe'+N.SUFFIX,))
    monkeypatch.setattr(N, 'CALCULATOR', None)
    monkeypatch.setattr(recdata, 'CALCULATOR_MODULES', ('probe',))
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    cache = tmp_path/'cache'
    assert not N.activate(cache)
    inputs = N.hashes(root, N.INPUTS)
    path = N.location(inputs, cache)
    path.mkdir(parents=True)
    # This check exercises manifest validation only; real extensions have a
    # separate subprocess import/output check after the native build.
    artifact = path/N.FILES[0]
    artifact.write_bytes(b'fixture')
    identity = recdata._calculator_identity([root/'probe.py', root/'calib/difficulty.json'])
    manifest = dict(abi=N.ABI, inputs=inputs, artifacts=N.hashes(path, N.FILES), calculator=identity)
    original_path = sys.path.copy()
    for field, bad in (('abi', 'other-python'), ('inputs', {}), ('artifacts', {}),
                       ('calculator', '000000000000')):
        (path/'manifest.json').write_text(json.dumps(dict(manifest, **{field: bad})))
        assert not N.activate(cache)
        assert N.CALCULATOR is None and sys.path == original_path
    (path/'manifest.json').write_text(json.dumps(manifest))
    artifact.write_bytes(b'changed')
    assert not N.activate(cache)
    artifact.write_bytes(b'fixture')
    (root/'probe.py').write_text('value = 2\n')
    assert not N.activate(cache)
    (root/'probe.py').write_text('value = 1\n')
    with monkeypatch.context() as scope:
        scope.setitem(sys.modules, 'probe', object())
        assert not N.activate(cache)
    assert N.activate(cache)
    assert N.CALCULATOR == identity and sys.path[0] == str(path)
    (root/'probe.py').write_text('value = 3\n')
    assert N.activate(cache) and N.CALCULATOR == identity

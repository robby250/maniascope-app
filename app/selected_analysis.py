"""Crash boundary for optional/native selected-map feature calculation.

The viewer's chart-only Python analysis remains independent. A native PP helper
aborting on one selected file must not take GTK and the score/event owner down.
Only a cache miss starts this low-priority child; the newest request replaces
pending old selections in recommend.Worker.
"""
import json
import os
import subprocess
import sys
import math
from pathlib import Path
import re
import threading


FROZEN = bool(getattr(sys, "frozen", False))    # packaged app: code cannot change under it
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # Windows: no console flash per selection


def _child(*args):
    """Command for a crash-isolated child: this file, or the packaged app's own entry."""
    if FROZEN:
        return [sys.executable, "--selected-analysis", *args]
    return [sys.executable, "-X", "faulthandler", __file__, *args]


def preserve_runtime(directory=None):
    """Keep the loaded calculator's helper usable after an on-disk update."""
    if FROZEN:
        return None
    import shutil
    import tempfile
    import recdata
    import native_backend
    source, root = native_backend.APP, native_backend.ROOT
    base = Path(directory) if directory is not None else Path(recdata.REC_DIR)/'runtime'
    calculator = recdata.calc_id()
    if previous_helper(calculator, base):
        return base/calculator
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=base) as temp:
        target = Path(temp)/calculator
        target.mkdir()
        # The snapshot is flat (modules beside native/ and calib/), as a packaged build.
        files = [(p, p.name) for p in source.glob('*.py') if not p.name.startswith('test_')]
        files += [(p, p.relative_to(root)) for p in native_backend.sources(root) + [root/'calib/difficulty.json']]
        for path, rel in files:
            dest = target/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        copied = recdata._calculator_identity(native_backend.identity_files(target))
        if copied != calculator:
            raise OSError('Calculator changed while preserving its runtime')
        target.rename(base/calculator)
    return base/calculator


def previous_helper(calculator, directory=None):
    """An already-open viewer can finish requests using its exact saved code.

    Only a hex calculator ID beneath our own runtime snapshots is accepted.
    The archived helper independently validates its source identity on startup.
    This never replaces or restarts the viewer itself.
    """
    if not isinstance(calculator,str) or not re.fullmatch(r'[0-9a-f]{12}',calculator):
        return None
    if directory is None:
        import recdata
        directory=Path(recdata.REC_DIR)/'runtime'
    helper=Path(directory)/calculator/'selected_analysis.py'
    return helper if helper.is_file() and helper.resolve()!=Path(__file__).resolve() else None


_helper = None              # (process, calculator): one long-lived child serves every selection
_helper_lock = threading.Lock()


def calculate(path, rate, calculator):
    """Features of one chart at one rate, from the long-lived helper. A new interpreter per selection
    cost ~0.15 s on Linux and a whole packaged-app start on Windows. If the helper dies (a native
    crash on this chart) or refuses the calculator, a one-shot child repeats the request and reports."""
    with _helper_lock:
        try:
            return _served(path, rate, calculator)
        except _HelperGone:
            pass
    return _once(path, rate, calculator)


class _HelperGone(Exception):
    pass


def _served(path, rate, calculator):
    global _helper
    if _helper is None or _helper[1] != calculator or _helper[0].poll() is not None:
        _stop()
        process = subprocess.Popen(_child("--serve", calculator), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, text=True, bufsize=1)
        _helper = (process, calculator)
    process = _helper[0]
    reply = []
    try:
        process.stdin.write(json.dumps([path, rate]) + "\n")
        process.stdin.flush()
        reader = threading.Thread(target=lambda: reply.append(process.stdout.readline()), daemon=True)
        reader.start()
        reader.join(60)
    except (OSError, ValueError):
        pass
    if not reply or not reply[0]:
        _stop()
        raise _HelperGone
    answer = json.loads(reply[0])
    if "error" in answer:
        raise RuntimeError("Selected chart analysis failed: " + answer["error"])
    return {float(key): value for key, value in answer["feats"].items()}


def _stop():
    global _helper
    if _helper is not None:
        _helper[0].kill()
        _helper[0].wait()
        _helper = None


def _once(path, rate, calculator):
    result = subprocess.run(_child(path, repr(rate), calculator), stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=60)
    if result.returncode:
        reason = result.stderr.strip()[-1600:] or f"child exited with status {result.returncode}"
        raise RuntimeError("Selected chart analysis failed: "+reason)
    data = json.loads(result.stdout)
    return {float(key): value for key, value in data.items()}


def performance(jobs):
    """Batch missing score PP without putting native scoring in a GTK process.

    A child emits each completed result separately. A native abort on a later
    score does not discard earlier results or terminate the viewer/replay owner.
    """
    if not jobs:
        return {}
    try:
        result = subprocess.run(_child("--pp"), creationflags=NO_WINDOW,
                                input=json.dumps(jobs), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=120)
        stdout, stderr, status = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and waits for its child on timeout. Preserve
        # completed JSON lines just as for an abort later in the batch.
        stdout, stderr, status = exc.stdout or '', exc.stderr or '', 'timeout'
    if isinstance(stdout, bytes): stdout = stdout.decode('utf-8', errors='replace')
    if isinstance(stderr, bytes): stderr = stderr.decode('utf-8', errors='replace')
    output = {}
    wanted = {item['key'] for item in jobs}
    for line in stdout.splitlines():
        try:
            key,value = json.loads(line)
            if key in wanted and (value is None or isinstance(value,(int,float)) and not isinstance(value,bool)
                                  and math.isfinite(value) and value >= 0):
                output[key] = value
        except (ValueError,TypeError):
            continue
    if status:
        print(f"PP helper failed ({status}); completed scores retained: "+stderr[-1600:], file=sys.stderr, flush=True)
    return output


def child_main(argv):
    import paths
    if argv[0] == "--pp":
        paths.lower_priority()
        import rosu_pp_py as rp
        for item in json.load(sys.stdin):
            try:
                s=item['stats']
                value=rp.Performance(mods=item['mods'],lazer=True,n_geki=s[0],n300=s[1],n_katu=s[2],
                                     n100=s[3],n50=s[4],misses=s[5]).calculate(rp.Beatmap(path=item['path'])).pp
            except Exception:
                value=None
            print(json.dumps([item['key'],value],allow_nan=False),flush=True)
        return 0
    serve = argv[0] == "--serve"
    calculator = argv[1] if serve else argv[2]
    import recdata
    if recdata.calc_id() != calculator:
        archived=previous_helper(calculator)
        if archived is not None and not serve:      # an archived helper may predate --serve
            os.execv(sys.executable,[sys.executable,'-X','faulthandler',str(archived),*argv])
        raise SystemExit("Calculator changed during selection; reopen after deployment completes")
    paths.lower_priority()
    from analysis_cache import selected
    if serve:
        import traceback
        for line in sys.stdin:
            path, rate = json.loads(line)
            try:
                chart, analysis = selected.analyze(path, rate, calculator=calculator)
                answer = {"feats": recdata.chart_feats(path, [rate], analyses={rate: analysis})}
            except Exception as exc:
                answer = {"error": "".join(traceback.format_exception_only(exc)).strip()[-1600:]}
            print(json.dumps(answer, allow_nan=False), flush=True)
        return 0
    rate = float(argv[1])
    chart, analysis = selected.analyze(argv[0], rate, calculator=argv[2])
    print(json.dumps(recdata.chart_feats(argv[0], [rate], analyses={rate:analysis}), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(child_main(sys.argv[1:]))

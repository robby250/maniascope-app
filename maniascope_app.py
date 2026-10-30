"""Packaged-app entry point (PyInstaller). From source, run skillsets.py as before.

The frozen executable is also its own crash-isolated helper: `--selected-analysis ARGS`
runs selected_analysis.child_main, and multiprocessing's spawn children re-enter here.
"""
import multiprocessing
import os
import sys


def self_test():
    """CI/support check without a game: GTK, calculator identity, a full analysis, dan table."""
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk
    assert Gtk.init_check()[0], "GTK cannot open a display"
    import dans
    import recdata
    import skill_calc
    here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    res = skill_calc.compute(skill_calc.parse_osu(os.path.join(here, "packaging", "selftest.osu")), 1.0)
    assert 1 < res["scores"]["overall"] < 20, res["scores"]["overall"]
    assert dans._table(), "dan table missing or built by another calculator"
    import selected_analysis
    got = selected_analysis.calculate(os.path.join(here, "packaging", "selftest.osu"), 1.0, recdata.calc_id())
    assert abs(got[1.0]["overall"] - res["scores"]["overall"]) < .5
    native = os.path.splitext(skill_calc.__file__)[1] in (".pyd", ".so")
    print("self-test OK", recdata.calc_id(), round(res["scores"]["overall"], 2), "native" if native else "python", flush=True)
    return 0

def log_to_file():
    """The windowed exe has no console: prints, tracebacks, faulthandler and GTK warnings go to
    maniascope.log (previous run: maniascope.prev.log) in the cache folder."""
    import platform
    import time
    import paths
    log = os.path.join(paths.CACHE, "maniascope.log")
    try:
        os.makedirs(paths.CACHE, exist_ok=True)
        try:
            os.replace(log, os.path.join(paths.CACHE, "maniascope.prev.log"))
        except OSError:
            pass  # first run, or another instance still has it open: append
        fh = open(log, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return
    for fd in (1, 2):
        os.dup2(fh.fileno(), fd)  # C-level GLib/GTK output too
    sys.stdout = sys.stderr = fh
    here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    try:
        calc = open(os.path.join(here, "calc_id.txt")).read().strip()
    except OSError:
        calc = "?"
    print(f"[{time.strftime('%F %T')}] ManiaScope calc {calc}, {platform.platform()}, Python {platform.python_version()}, "
          f"{sys.executable}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--selected-analysis":
        import selected_analysis
        raise SystemExit(selected_analysis.child_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "--webmaps":
        import webmaps
        raise SystemExit(webmaps.main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        raise SystemExit(self_test())
    log_to_file()
    import skillsets
    skillsets.main()

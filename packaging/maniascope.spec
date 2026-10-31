# PyInstaller spec (one-folder). Build from the repository root:
#   python -c "import sys; sys.path.insert(0, 'app'); import recdata; print(recdata.calc_id())" > calc_id.txt
#   pyinstaller packaging/maniascope.spec
# Windows CI adds the GTK3 runtime (gvsbuild) and node+realm; see .github/workflows/windows.yml.
import glob
import os
import sys

root = os.path.abspath(os.path.join(SPECPATH, ".."))
R = lambda *p: os.path.join(root, *p)
datas = [(R("calc_id.txt"), "."), (R("app", "lazer_scores.js"), "."), (R("assets/icon.png"), "."),
         (R("calib/difficulty.json"), "calib"), (R("calib/dans.json"), "calib"),
         (R("packaging/selftest.osu"), "packaging"), (R("app", "theme"), "theme"), (R("app", "overlay"), "overlay"),
         (R("native/src/personal.rs"), "native/src")]   # recommend.predictor_id hashes the personal fit
# Calculator sources stay readable: the analysis helper and diagnostics hash/parse them.
# Flat bundle: modules beside calib/ and native/ (native_backend.tree_root).
sources = [p for p in glob.glob(R("app", "*.py")) if not os.path.basename(p).startswith("test_")]
datas += [(p, ".") for p in sources]
for extra in ("realmjs", "node"):          # staged by CI (Windows); optional elsewhere
    if os.path.isdir(R(extra)):
        datas.append((R(extra), extra))

a = Analysis([R("app", "maniascope_app.py")], pathex=[R("app")], datas=datas,
             hiddenimports=[os.path.splitext(os.path.basename(p))[0] for p in sources],
             excludes=["pytest", "tkinter"],
             # The UI uses a few symbolic icons; the gi hook would otherwise copy every system theme.
             hooksconfig={"gi": {"icons": ["Adwaita", "hicolor"], "themes": ["Adwaita"], "languages": ["en_US"]}})
# The Rust calculator (build_windows_wine.sh cross-builds it); native_backend imports it by name.
a.binaries += [("maniascope_native.pyd", R("native", "maniascope_native.pyd"), "EXTENSION")]
# abi3 modules link the stable-ABI python3.dll, which PyInstaller doesn't collect (only python3XX.dll).
a.binaries += [("python3.dll", os.path.join(sys.base_prefix, "python3.dll"), "BINARY")]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ManiaScope", console=False,
          icon=R("packaging/icon.ico") if os.path.exists(R("packaging/icon.ico")) else None)
coll = COLLECT(exe, a.binaries, a.datas, name="ManiaScope")

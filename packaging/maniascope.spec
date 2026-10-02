# PyInstaller spec (one-folder). Build from the repository root:
#   python -c "import recdata; print(recdata.calc_id())" > calc_id.txt
#   pyinstaller packaging/maniascope.spec
# Windows CI adds the GTK3 runtime (gvsbuild) and node+realm; see .github/workflows/windows.yml.
import glob
import os

root = os.path.abspath(os.path.join(SPECPATH, ".."))
R = lambda *p: os.path.join(root, *p)
datas = [(R("calc_id.txt"), "."), (R("lazer_scores.js"), "."), (R("icon.png"), "."),
         (R("calib/difficulty.json"), "calib"), (R("calib/dans.json"), "calib"),
         (R("packaging/selftest.osu"), "packaging"), (R("theme"), "theme")]
# Calculator sources stay readable: the analysis helper and diagnostics hash/parse them.
sources = [p for p in glob.glob(R("*.py")) if not os.path.basename(p).startswith("test_")]
datas += [(p, ".") for p in sources]
for extra in ("realmjs", "node"):          # staged by CI (Windows); optional elsewhere
    if os.path.isdir(R(extra)):
        datas.append((R(extra), extra))

a = Analysis([R("maniascope_app.py")], pathex=[root], datas=datas,
             hiddenimports=[os.path.splitext(os.path.basename(p))[0] for p in sources],
             excludes=["pytest", "tkinter"],
             # The UI uses a few symbolic icons; the gi hook would otherwise copy every system theme.
             hooksconfig={"gi": {"icons": ["Adwaita", "hicolor"], "themes": ["Adwaita"], "languages": ["en_US"]}})
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ManiaScope", console=False,
          icon=R("packaging/icon.ico") if os.path.exists(R("packaging/icon.ico")) else None)
coll = COLLECT(exe, a.binaries, a.datas, name="ManiaScope")

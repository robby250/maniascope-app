#!/bin/bash
# Windows one-folder build on Linux through Wine (same steps as windows-workflow.yml; no CI or
# `workflow` token scope needed). Verified 2026-10-01 with wine-staging 11.14: source + frozen
# self-test, realm load, GUI start under Xvfb.
#   packaging/build_windows_wine.sh PUBLIC_CHECKOUT WORKDIR   → WORKDIR/ManiaScope-windows.zip
set -euo pipefail
SRC=$(realpath "$1"); W=$(realpath -m "$2"); mkdir -p "$W/dl"
export WINEPREFIX="$W/prefix" WINEDEBUG=-all
C="$WINEPREFIX/drive_c"
run() { wine "$@" </dev/null >"$W/dl/last.log" 2>&1 || { tail -20 "$W/dl/last.log"; exit 1; }; }  # a pipe would wait for wineserver
cd "$W/dl"
# Initialise the prefix before anything creates drive_c (else "could not load kernel32.dll"); no Mono/Gecko prompt.
[ -e "$WINEPREFIX/system.reg" ] || { WINEDLLOVERRIDES="mscoree,mshtml=" wineboot -i </dev/null >/dev/null 2>&1; wineserver -w; }
[ -e "$C/py/python.exe" ] || { curl -sLo py.nupkg https://www.nuget.org/api/v2/package/python/3.14.0
  rm -rf nug && mkdir nug && (cd nug && unzip -q ../py.nupkg) && mkdir -p "$C" && cp -r nug/tools "$C/py"; }
[ -d "$C/gtk" ] || { curl -sLo gtk.zip https://github.com/wingtk/gvsbuild/releases/download/2026.8.0/GTK3_Gvsbuild_2026.8.0_x64.zip
  mkdir -p "$C/gtk" && unzip -q gtk.zip -d "$C/gtk"; }
[ -d "$C/node" ] || { curl -sLo node.zip https://nodejs.org/dist/v22.20.0/node-v22.20.0-win-x64.zip
  unzip -q node.zip && mv node-v22.20.0-win-x64 "$C/node"; }
run 'C:\py\python.exe' -m pip install -q $(cd "$C/gtk/wheels" && ls | sed 's#^#C:\\gtk\\wheels\\#') numpy rosu-pp-py pyinstaller
rm -rf "$C/build" && git clone -q "$SRC" "$C/build" && cd "$C/build"
mkdir -p node realmjs && cp "$C/node/node.exe" node/
(cd realmjs && WINEPATH='C:\node' run 'C:\node\node.exe' 'C:\node\node_modules\npm\bin\npm-cli.js' init -y \
  && WINEPATH='C:\node' run 'C:\node\node.exe' 'C:\node\node_modules\npm\bin\npm-cli.js' install realm@12)
rm -rf realmjs/node_modules/realm/prebuilds/{android,apple}          # 930 MB of mobile static libraries
export WINEPATH='C:\gtk\bin' GI_TYPELIB_PATH='C:\gtk\lib\girepository-1.0'
wine 'C:\py\python.exe' -c "import recdata; print(recdata.calc_id())" </dev/null >calc_id.txt 2>/dev/null
[ "$(tr -d "\r" <calc_id.txt)" = "$(cat packaging/calc_id.expected)" ] || { echo "calc id $(cat calc_id.txt) != expected"; exit 1; }
run 'C:\py\python.exe' maniascope_app.py --self-test
# Compiled calculator (~2.7× faster): Cython C made here, mingw-w64 against the bundled Python's DLL and
# its UCRT (the default msvcrt would mix two C runtimes). Outputs match plain Python within 1e-9.
[ -x "$W/venv/bin/cython" ] || { python3 -m venv "$W/venv" && "$W/venv/bin/pip" -q install Cython==3.3.0; }
x86_64-w64-mingw32-gcc -dumpspecs | sed 's/-lmsvcrt/-lucrt/g' >"$W/dl/ucrt.specs"
mkdir -p native
for m in $(python3 -c "import recdata; print(*recdata.CALCULATOR_MODULES)"); do echo "$m"; done | xargs -P "$(nproc)" -I{} sh -c \
  '"$0/venv/bin/cython" -3 -X annotation_typing=True -X infer_types=True {}.py -o native/{}.c &&
   x86_64-w64-mingw32-gcc -specs="$0/dl/ucrt.specs" -D_UCRT -O2 -g0 -shared -DNDEBUG -w -I"$1/py/include" native/{}.c \
     -L"$1/py" -lpython314 -static-libgcc -o native/{}.cp314-win_amd64.pyd' "$W" "$C"
run 'C:\py\python.exe' -m PyInstaller --noconfirm packaging/maniascope.spec
(cd dist/ManiaScope && run ManiaScope.exe --self-test)
grep -q "self-test OK.*native" "$W/dl/last.log"
wineserver -k || true
rm -f "$W/ManiaScope-windows.zip" && (cd dist && zip -qr9 "$W/ManiaScope-windows.zip" ManiaScope)
echo "$W/ManiaScope-windows.zip"

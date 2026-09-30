#!/usr/bin/env bash
# One-time setup for ManiaScope: installs the application-menu launcher.
#
# Requirements: python3 with PyGObject (GTK 3) + pycairo
#   Debian/Ubuntu/Mint: sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0
# No compiler/toolchain needed — ManiaScope is pure Python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing desktop launcher…"
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
sed "s#@HERE@#$HERE#g" "$HERE/maniascope.desktop.in" > "$APPS/maniascope.desktop"
chmod +x "$APPS/maniascope.desktop"
update-desktop-database "$APPS" 2>/dev/null || true

TOSU_DIR="$HOME/.local/share/maniascope/tosu"
if [ ! -x "$TOSU_DIR/tosu" ]; then
  read -r -p "Download tosu (needed for automatic rate tracking, ~40 MB)? [y/N] " yn
  if [[ "$yn" =~ ^[Yy] ]]; then
    mkdir -p "$TOSU_DIR"
    url="$(curl -fsSL https://api.github.com/repos/tosuapp/tosu/releases/latest \
           | grep -o 'https://[^"]*tosu-linux-[^"]*\.zip' | head -1)"
    curl -fL -o "$TOSU_DIR/tosu.zip" "$url" && unzip -o -q "$TOSU_DIR/tosu.zip" -d "$TOSU_DIR" \
      && rm "$TOSU_DIR/tosu.zip" && chmod +x "$TOSU_DIR/tosu"
  fi
fi

REALM_DIR="$HOME/.local/share/maniascope/realmjs"
if [ ! -d "$REALM_DIR/node_modules/realm" ] && command -v npm >/dev/null; then
  read -r -p "Install Realm JS (lets lazer_scores.py import your local scores, ~150 MB)? [y/N] " yn
  if [[ "$yn" =~ ^[Yy] ]]; then
    mkdir -p "$REALM_DIR" && (cd "$REALM_DIR" && npm init -y >/dev/null && npm install realm@12)
  fi
fi

if ! python3 -c "import rosu_pp_py" 2>/dev/null; then
  read -r -p "Install rosu-pp-py (pp calculation for Next recommendations, ~3 MB)? [y/N] " yn
  if [[ "$yn" =~ ^[Yy] ]]; then
    python3 -m pip install --user --break-system-packages rosu-pp-py
  fi
fi

echo
echo "Done. Launch 'ManiaScope' from your application menu, or run ./run.sh"

# ManiaScope

osu!mania difficulty viewer and practice playlists for **osu!lazer**. It follows the map and rate you
select in lazer and shows its difficulty, skills, dan equivalent and NPS. Its playlists suggest what
to play next:

- **PP**: maps where you are likely to beat a best score. It pushes harder as you beat scores and eases
  off after bad plays.
- **NPS**: dense practice at your chosen accuracy target (default 94%), including website maps you
  don't have yet (download suggested).
- **Skills**: practice for chosen skills (jacks, LN, tech, …).

## Windows

1. Download `ManiaScope-windows.zip` from Releases and unzip it anywhere.
2. Run `ManiaScope.exe`. On first start it downloads **tosu** (to follow lazer's rate) and the
   population data for predictions (~140 MB).
3. Optional: in ⋯ → **osu! profile…**, enter your username. Your public top-100 plays are then used as
   bests (no password or API key).
4. Optional: in the NPS tab → Settings → **Website charts**, pick a ★ range and press Download. This
   runs in the background at low priority and fetches only chart text, never audio.

Your lazer scores are read locally from lazer's database (read-only copy).

## Linux

Runs from source with your system's Python. Open a terminal and do these steps once.

**1. Install the dependencies** (pick your distro):

| Distro | Command |
|---|---|
| Fedora | `sudo dnf install git python3-gobject gtk3 python3-cairo python3-numpy python3-pip` |
| Debian / Ubuntu / Mint / Pop!_OS | `sudo apt install git python3-gi python3-gi-cairo gir1.2-gtk-3.0 python3-numpy python3-pip` |
| Arch / Manjaro / EndeavourOS | `sudo pacman -S git python-gobject gtk3 python-cairo python-numpy python-pip` |

**2. Install the pp calculator** (same on every distro):

```
python3 -m pip install --user rosu-pp-py
```

If pip answers `externally-managed-environment`, run it again with `--break-system-packages` at the end.

**3. Download ManiaScope and set it up:**

```
git clone https://github.com/robby250/maniascope-app.git
cd maniascope-app
./setup.sh
```

`setup.sh` adds ManiaScope to your application menu and asks about optional downloads:

- **tosu** (say yes): lets ManiaScope follow the rate you pick in lazer.
- **Realm JS** (optional, needs `nodejs` and `npm`): imports your local lazer scores. It is only offered when npm is installed.
- **rosu-pp-py**: skip if you already did step 2.

**4. Start it** from the application menu (ManiaScope), or with `./run.sh` in that folder. On first start
it downloads the population data for predictions (~140 MB).

osu!lazer is found automatically whether it is the AppImage (`~/.local/share/osu`) or the Flatpak. For
another location start it as `LAZER_DATA=/path/to/osu ./run.sh`.

**Update later:** `cd maniascope-app && git pull`, then restart ManiaScope.

**If it doesn't start:** the log is `~/.cache/maniascope/run.log`. `No module named 'gi'` means step 1
is missing. `No module named 'numpy'` means `python3-numpy` from step 1 is missing.

## Notes

- Difficulty ratings are provisional; dan labels are the community courses rated with the same
  calculator.
- Website charts come from osu.direct, with ppy as a slow fallback. Beatmaps belong to their mappers.
- osu! is a trademark of ppy Pty Ltd; ManiaScope is an unaffiliated third-party tool. MIT licence.

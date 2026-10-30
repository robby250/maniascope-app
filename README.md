# ManiaScope

osu!mania difficulty viewer and practice playlists for **osu!lazer** and **osu!stable**. It follows the map
and rate you select in the game and shows its difficulty, skills, dan equivalent and NPS. Its playlists suggest
what to play next:

- **PP**: maps where you are likely to beat a best score. It pushes harder as you beat scores and eases
  off after bad plays.
- **NPS**: dense practice at your chosen accuracy target (default 95%), including website maps you
  don't have yet (download suggested).
- **Skills**: practice for chosen skills (jacks, LN, tech, …).

## Windows

1. Download `ManiaScope-windows.zip` from Releases and unzip it anywhere.
2. Run `ManiaScope.exe`. On first start it downloads **tosu** (to follow lazer's rate) and the
   population data for predictions (~175 MB).
3. Optional: in ⋯ → **osu! profile…**, enter your username. Your public top-100 plays are then used as
   bests (no password or API key).
4. Optional: in the playlist window, NPS tab → ⚙ → **Website charts**, pick a ★ range and press Download. This
   runs in the background at low priority and fetches only chart text, never audio.

Your lazer scores are read locally from lazer's database (read-only copy).

## osu!stable

ManiaScope switches to whichever osu! tosu sees running. With stable:

- its library and your scores come from `osu!.db` and `scores.db` (read-only). Stable rewrites `osu!.db` when it
  closes, so maps added in a session join the playlists after you next close stable (selecting them works at once);
- playlists offer only HT (0.75×), NM and DT (1.5×), the rates stable has;
- accuracies are shown as stable shows them (MAX counts 300, so about half a percent higher than lazer's);
- your local replays (`Data/r`) of tracked plays are archived as lazer's are, with the pauses tosu saw (stable writes
  them when it closes, so they are picked up the next time it starts);
- a chart has the same identity in both games, so your lazer history keeps counting.

Stable is found in `%LOCALAPPDATA%\osu!` (Windows) or osu-winello's folder (Linux); elsewhere, start ManiaScope
with `STABLE_DATA=<folder with osu!.db>`. On Linux tosu needs `kernel.yama.ptrace_scope=0` (Fedora's default) to
read a Wine game.

## Linux

Runs from source with your system's Python. Open a terminal and do these steps once.

**1. Install the system packages** (pick your distro):

| Distro | Command |
|---|---|
| Fedora | `sudo dnf install git python3-gobject gtk3 python3-cairo python3-numpy python3-pip unzip` |
| Debian / Ubuntu / Mint / Pop!_OS | `sudo apt install git python3-gi python3-gi-cairo gir1.2-gtk-3.0 python3-numpy python3-pip unzip` |
| Arch / Manjaro / EndeavourOS | `sudo pacman -S git python-gobject gtk3 python-cairo python-numpy python-pip unzip` |

**2. Download ManiaScope and set it up:**

```
git clone https://github.com/robby250/maniascope-app.git
cd maniascope-app
./setup.sh
```

`setup.sh` installs everything else for your user only (no sudo) and says what it is doing:

- the pp calculator (rosu-pp-py);
- the score reader, which reads your lazer scores and installed maps (Node.js and Realm; downloaded if
  you don't have Node);
- the difficulty calculator (prebuilt for each release; on a newer `git pull` it is built with
  [Rust](https://rustup.rs) if you have it);
- tosu (it asks; say yes), which lets ManiaScope see the rate you pick in lazer;
- the application-menu entry.

**3. Start it** from the application menu (ManiaScope), or with `./run.sh` in that folder. The first
start downloads the population data for predictions (~175 MB, progress shown at the bottom of the Playlist
window) and analyses your maps once in the background (it shows how many and how long).

osu!lazer is found automatically whether it is the AppImage (`~/.local/share/osu`) or the Flatpak. For
another location start it as `LAZER_DATA=/path/to/osu ./run.sh`.

**Update later:** `cd maniascope-app && git pull && ./setup.sh`, then restart ManiaScope.

**If it doesn't start:** the log is `~/.cache/maniascope/run.log`. Run `./setup.sh` again: it checks
every requirement and says which one is missing.

## Notes

- Difficulty ratings are provisional; dan labels are the community courses rated with the same
  calculator.
- Website charts come from osu.direct, with ppy as a slow fallback. Beatmaps belong to their mappers.
- osu! is a trademark of ppy Pty Ltd; ManiaScope is an unaffiliated third-party tool. MIT licence.

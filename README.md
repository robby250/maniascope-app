# ManiaScope

> **Warning:** untested on Windows, osu!stable and 4K. There could be a lot of bugs, especially in those cases.

osu!mania difficulty viewer and practice playlists for **osu!lazer** and **osu!stable**. It follows the map
and rate you select in the game and shows its difficulty, skills, dan equivalent and NPS. Its playlists suggest
what to play next:

- **PP**: maps where you are likely to beat a best score. It pushes harder as you beat scores and eases
  off after bad plays.
- **NPS**: dense practice at your chosen accuracy target (default 95%), including website maps you
  don't have yet (download suggested).
- **Skills**: practice for chosen skills (jacks, LN, tech, …).

NPS and Skills can also offer maps converted to the keymode you are playing (⚙ → **Converts**): off by default, built
with [Torosu](https://github.com/Manolo-dl/Torosu) and imported into the game when one comes up.

## Windows

1. Download `ManiaScope.exe` from Releases and put it anywhere.
2. Run it. On first start it downloads the population data for predictions (~200 MB).
3. Optional: in ⋯ → **osu! profile…**, enter your username. Your public top-100 plays are then used as
   bests (no password or API key).
4. Optional: in the playlist window, NPS tab → ⚙ → **Website charts**, pick a ★ range and press Download. This
   runs in the background at low priority and fetches only chart text, never audio.

Your lazer scores are read locally from lazer's database (read-only copy).

## osu!stable

ManiaScope switches to whichever osu! is running. With stable:

- its library and your scores come from `osu!.db` and `scores.db` (read-only). Stable rewrites `osu!.db` when it
  closes, so maps added in a session join the playlists after you next close stable (selecting them works at once);
- playlists offer only HT (0.75×), NM and DT (1.5×), the rates stable has;
- accuracies are shown as stable shows them (MAX counts 300, so about half a percent higher than lazer's);
- your local replays (`Data/r`) of tracked plays are archived as lazer's are, with the pauses seen during play (stable writes
  them when it closes, so they are picked up the next time it starts);
- a chart has the same identity in both games, so your lazer history keeps counting.

Stable's folder is found from the running game, whatever its location. On Linux reading a Wine game needs
`kernel.yama.ptrace_scope=0` (Fedora's default).

## Linux

1. Download `maniascope-linux-x86_64` from Releases and put it where it can stay.
2. Make it runnable and add it to the application menu:

   ```
   chmod +x maniascope-linux-x86_64
   ./maniascope-linux-x86_64 --install-desktop
   ```

3. Start **ManiaScope** from the menu. The first start downloads the population data for predictions (~200 MB, progress shown at the bottom of the Playlist window), then
   analyses your maps once in the background (it shows how many and how long).

osu!lazer is found automatically whether it is the AppImage (`~/.local/share/osu`) or the Flatpak. For another location
start it as `LAZER_DATA=/path/to/osu ./maniascope-linux-x86_64`.

**Update later:** replace the file with the new one from Releases.

**If it doesn't start:** the log is `~/.cache/maniascope/maniascope.log`.

## Notes

- Difficulty ratings are provisional; dan labels are the community courses rated with the same
  calculator.
- Website charts come from osu.direct, with ppy as a slow fallback. Beatmaps belong to their mappers.
- osu! is a trademark of ppy Pty Ltd; ManiaScope is an unaffiliated third-party tool. MIT licence, except
  `crates/ms-osumem`: reading the game is ported from [tosu](https://github.com/tosuapp/tosu) and stays LGPL-3.0.
  ManiaScope no longer runs or downloads tosu. If you run your own tosu, ManiaScope can add its page there
  (⋯ → Settings).

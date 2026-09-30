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

Needs `python3-gi python3-gi-cairo gir1.2-gtk-3.0` and `pip install numpy rosu-pp-py`. Then
`./setup.sh` and `./run.sh`.

## Notes

- Difficulty ratings are provisional; dan labels are the community courses rated with the same
  calculator.
- Website charts come from osu.direct, with ppy as a slow fallback. Beatmaps belong to their mappers.
- osu! is a trademark of ppy Pty Ltd; ManiaScope is an unaffiliated third-party tool. MIT licence.

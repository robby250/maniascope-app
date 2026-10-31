#!/usr/bin/env python3
"""
ManiaScope — glanceable osu!mania difficulty for the map selected in osu!lazer.

  * selection + live rate: a local tosu helper (http://127.0.0.1:24050/json/v2)
    reports the selected chart's file and the selected mods while still in song
    select, so custom rates such as 0.98× are followed exactly (Auto).
  * fallback without tosu: tail lazer's log and resolve the map through
    lazer_index.py. That source has NO rate, and the UI says so.
  * "Open .osu" analyses a local file and pins it until "Follow lazer".

Requires python3-gi (GTK 3) + pycairo; pure stdlib otherwise.
"""
import os
# One look on every desktop and on Windows: the bundled Breeze Dark theme and icons (theme/README.md).
# GLib reads XDG_DATA_DIRS once, so this precedes GTK; frozen builds already list their share/ there.
_THEME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme")
os.environ["XDG_DATA_DIRS"] = os.pathsep.join(filter(None, [
    _THEME_DIR, os.environ.get("XDG_DATA_DIRS") or ("" if os.name == "nt" else "/usr/local/share:/usr/share")]))
import gi  # noqa: E402
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gdk, Gtk, GLib, Pango, PangoCairo  # noqa: E402
import cairo  # noqa: E402

import functools  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


import feedback  # noqa: E402
import hashlib
import recdata
import lazer_index  # noqa: E402
import recommend  # noqa: E402
import nps as nps_playlist  # noqa: E402
import navigation
import dans  # noqa: E402
import skill_calc  # noqa: E402
import skill_practice

import paths  # noqa: E402
LAZER_DATA = paths.lazer_data()
LAZER_LOGS = os.path.join(LAZER_DATA, "logs")
LAZER_FILES = os.path.join(LAZER_DATA, "files")
TOSU_URL = os.environ.get("TOSU_URL", "http://127.0.0.1:24050/json/v2")
import tools_setup  # noqa: E402
TOSU_BIN = os.path.expanduser(os.environ.get("TOSU_BIN", tools_setup.tosu_bin()))
# The Linux build reads ~/.config/tosu; the Windows build reads tosu.env beside its executable.
TOSU_ENV = os.path.join(os.path.dirname(TOSU_BIN), "tosu.env") if paths.WINDOWS else \
    os.path.expanduser("~/.config/tosu/tosu.env")

# Only the patched tosu (patches/tosu-loop-waits.patch) refreshes song select faster than its 150 ms POLL_RATE:
# 100 Hz of requests against an official build reads the same state and costs both processes CPU.
TOSU_SELECT_POLL = 0.01 if os.path.exists(os.path.join(os.path.dirname(TOSU_BIN), "procfix-build.json")) else 0.1

CACHE_DIR = paths.CACHE
CRASH_LOG = os.path.join(CACHE_DIR, "crash.log")
UI_FILE = os.path.join(paths.CONFIG, "ui.json")

BAR_FULL = 10.0   # fixed scale for the skill bars: comparable across maps
# Developer tools (saved-map comparisons, arbitrary .osu, prediction history) stay out of
# release builds (user 2026-10-01); running from source or MANIASCOPE_DEBUG=1 shows them.
DEBUG = os.environ.get("MANIASCOPE_DEBUG") == "1" or not getattr(sys, "frozen", False)
def first_run_prompt(db):
    """Standalone builds ask once for the osu! username (friends have no other score source set up)."""
    if not getattr(sys, "frozen", False) or recdata.kv_get(db, "first_run_asked"):
        return False
    recdata.kv_set(db, "first_run_asked", True)
    return True


def tracking_switch(db):
    """Only source/debug runs get the pause switch (user 2026-10-01). Release builds have no way back
    on, so an old pause is closed instead of excluding every later play."""
    if DEBUG:
        return True
    recdata.set_tracking(db, True)
    return False


DISPLAY_DEFAULTS = {"rating": True, "nps": True, "dan": True, "ln": True, "timeline": True, "skills": 3}
# NPS badge: WoW item-rarity colours (poor, uncommon, rare, epic, legendary) centred on each 10 NPS
# band and blended between them; from 20 NPS the digits run as a gradient that widens to 10 NPS by 30
NPS_SPECTRUM = ((5, "#9d9d9d"), (15, "#1eff00"), (25, "#0070dd"), (35, "#a335ee"), (45, "#ff8000"),
                (55, "#ff3c00"))


def nps(chart, rate):
    """Notes per second of played time, first to last note (None under 2 notes)."""
    return recdata.raw_nps(chart, rate)


def _mix(a, b, f):
    a, b = int(a[1:], 16), int(b[1:], 16)
    return "#" + "".join(f"{round(((a >> s) & 255) * (1 - f) + ((b >> s) & 255) * f):02x}" for s in (16, 8, 0))


class GradLabel(Gtk.DrawingArea):
    """Text filled with a smooth left→right colour gradient (Pango markup only colours whole glyphs)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._layout, self._cols, self._end, self._bg, self._pad = None, (), 0, None, 0
        self.connect("draw", self._draw)

    def set(self, markup="", cols=("#888888",), end=None, bg=None):
        """markup: sizes/weights only; cols: gradient stops (#rrggbb[aa]) spanning the first `end`
        chars (all if None); bg: colour of a rounded pill behind the text (lazer's star badge)."""
        self._layout = self.create_pango_layout("")
        self._layout.set_markup(markup, -1)
        self._cols, self._bg = cols, bg
        n = len(self._layout.get_text()) if end is None else end
        self._end = self._layout.index_to_pos(n).x / Pango.SCALE if n else 0
        w, h = self._layout.get_pixel_size()
        self._pad = round(h * 0.2) if bg else 0
        self.set_size_request(w + 2 * self._pad, h)
        self.queue_draw()

    def _draw(self, _w, cr):
        if not self._layout:
            return
        rgba = lambda c: [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [int(c[7:9] or "ff", 16) / 255]
        w, h = self._layout.get_pixel_size()
        if self._bg:
            r = h / 2
            cr.arc(r, r, r, math.pi / 2, 3 * math.pi / 2)
            cr.arc(w + 2 * self._pad - r, r, r, -math.pi / 2, math.pi / 2)
            cr.close_path()
            cr.set_source_rgba(*rgba(self._bg))
            cr.fill()
        g = cairo.LinearGradient(self._pad, 0, self._pad + max(1.0, self._end), 0)
        for i, c in enumerate(self._cols):
            g.add_color_stop_rgba(i / max(1, len(self._cols) - 1), *rgba(c))
        cr.move_to(self._pad, 0)
        PangoCairo.layout_path(cr, self._layout)
        cr.set_source(g)
        cr.fill()


def nps_parts(v):
    """→ GradLabel.set args: the gradient runs over the number, " NPS" takes its last colour."""
    text = f"{v:.1f}"
    lo = v - 10 * min(1.0, max(0.0, (v - 20) / 10))
    cols = tuple(c for x, c in NPS_SPECTRUM if lo < x < v)
    cols = (_sample(NPS_SPECTRUM, lo),) + cols + (_sample(NPS_SPECTRUM, v),)
    return (f'<span size="190%" weight="bold">{text}</span><span size="90%"> NPS</span>', cols, len(text))


# osu!lazer star badge (osu.Game/Graphics/OsuColour.cs): the pill takes STAR_SPECTRUM, its text is
# 75 % black below 6.5 ★, Orange1 up to 9 ★, then STAR_TEXT_SPECTRUM. From 10 ★ the text runs as a
# smooth gradient from 9 ★ yellow up to its own colour.
STAR_SPECTRUM = ((0.1, "#aaaaaa"), (0.1, "#4290fb"), (1.25, "#4fc0ff"), (2.0, "#4fffd5"), (2.5, "#7cff4f"),
                 (3.3, "#f6f05c"), (4.2, "#ff8068"), (4.9, "#ff4e6f"), (5.8, "#c645b8"), (6.7, "#6563de"),
                 (7.7, "#18158e"), (9.0, "#000000"))
STAR_TEXT_SPECTRUM = ((9.0, "#f6f05c"), (9.9, "#ff8068"), (10.6, "#ff4e6f"), (11.5, "#c645b8"), (12.4, "#6563de"))


def _sample(spec, x):
    if x <= spec[0][0]:
        return spec[0][1]
    for (a, ca), (b, cb) in zip(spec, spec[1:]):
        if x <= b:
            return _mix(ca, cb, (x - a) / (b - a) if b > a else 1.0)
    return spec[-1][1]


def rating_parts(v):
    """→ GradLabel.set args for the big number, drawn as lazer's star badge."""
    v = round(v, 2)
    if v < 6.5:
        cols = ("#000000bf",)
    elif v < 9.0:
        cols = ("#ffd966",)
    elif v < 10.0:
        cols = (_sample(STAR_TEXT_SPECTRUM, v),)
    else:
        cols = tuple(c for x, c in STAR_TEXT_SPECTRUM if x < v) + (_sample(STAR_TEXT_SPECTRUM, v),)
    return f'<span size="240%" weight="bold">{v:.2f}</span>', cols, None, _sample(STAR_SPECTRUM, v)


def log_exc(where, info=None):
    """Append the current exception traceback to the crash log and stderr."""
    detail = "".join(traceback.format_exception(*info)) if info else traceback.format_exc()
    msg = f"\n=== {where} @ {__import__('datetime').datetime.now():%F %T} ===\n{detail}"
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CRASH_LOG, "a", encoding="utf-8") as fh:
            fh.write(msg)
    except OSError:
        pass
    print(msg, file=__import__("sys").stderr, flush=True)


def lifecycle(message):
    # Low-volume evidence distinguishes a close/hide/signal from a native crash.
    print(f"[{time.strftime('%F %T')}] viewer pid={os.getpid()} {message}", flush=True)


rate_from_mods = feedback.rate_from_mods


def analyze(path, rate, sv=True, *, cancelled=None):
    from analysis_cache import selected
    return selected.analyze(path, round(rate, 4), sv, cancelled=cancelled)


# ---------------------------------------------------------------------------
# Observing lazer: tosu first, log tail as the rate-less fallback.
# ---------------------------------------------------------------------------
def _size(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _duration(s):
    m = int(s // 60)
    return ("under a minute" if m < 1 else f"{m} min" if m < 60 else
            f"{m // 60} h {m % 60} min" if m < 1440 else f"{m // 1440} d {m // 60 % 24} h")


class LazerWatcher(threading.Thread):
    """Emits an observation dict whenever the selection/rate/connection changes.

    {"source": "tosu"|"log"|None, "path": str|None, "rate": float|None,
     "note": str}  — rate None means unknown, never No Mod.
    """

    _LINE_RE = re.compile(r"Game-wide working beatmap updated to (.+?)\s*$")

    def __init__(self, on_obs, on_status, on_progress, on_score, on_attempt):
        super().__init__(daemon=True)
        self.on_obs, self.on_status, self.on_progress = on_obs, on_status, on_progress
        self.on_score, self._last_score, self.on_attempt = on_score, None, on_attempt
        self._play = self._done = None   # the attempt in gameplay / the one whose result comes next
        self._live = self._last_live = None   # playback ms while in gameplay, else None
        self._halt = threading.Event()
        self._last = None
        self._index = None
        self._log = (None, 0)       # (path, read position)
        self._log_display = None
        self._last_reindex = 0.0
        self._index_job = None
        self._needs_import = False
        self._disconnected = False
        self._stable_songs, self._stable_files = None, {}

    def stop(self):
        self._halt.set()

    def run(self):
        while not self._halt.is_set():
            try:
                self._live = None
                obs = self._poll_tosu()
                tosu = obs is not None
                if not tosu or not obs.get("path"):
                    self._needs_import = True
                    self._disconnected = True
                elif self._needs_import and self._play is None:
                    self._needs_import = False
                    self.on_attempt("import")       # missed results survive in lazer's database
                if not tosu:
                    obs = self._poll_log()
                if obs != self._last:
                    self._last = obs
                    self.on_obs(obs)
                if self._live != self._last_live:
                    self._last_live = self._live
                    self.on_progress(self._live)
            except Exception:
                log_exc("watcher")
                tosu = False
            # A cached chart is ready in milliseconds; 100 ms polling dominated
            # selection latency. Keep gameplay and unavailable-game polling slow.
            delay = (0.3 if self._live is not None else (TOSU_SELECT_POLL if obs.get("path") else 0.1)) if tosu else 1.0
            self._halt.wait(delay)

    # -- tosu ---------------------------------------------------------------
    def _poll_tosu(self):
        try:
            with urllib.request.urlopen(TOSU_URL, timeout=0.6) as resp:
                d = json.load(resp)
        except urllib.error.HTTPError:
            return {"source": "tosu", "path": None, "rate": None,
                    "note": "Waiting for osu!: start it and select a mania map"}
        except (OSError, ValueError):
            return None   # tosu not reachable → log fallback
        try:
            if d.get("error") or "beatmap" not in d:
                return {"source": "tosu", "path": None, "rate": None,
                        "note": "Waiting for osu!: start it and select a mania map"}
            # stable opens on an osu!standard map; converts are unsupported, so only mania charts are analysed
            mania = (d["beatmap"].get("mode") or {}).get("number", 3) == 3
            path = self._chart(d) if mania else None
            self._track(d["state"]["name"].lower(), d, path)
            if d["state"]["name"] == "resultScreen":
                self._note_score(d, path)
                if self._last and self._last.get("source") == "tosu":
                    return self._last   # a past score's mods are not the selection
            rate, note = rate_from_mods(d["play"].get("mods"))
            cs = feedback.constant_speed(d["play"].get("mods"))
            if d["state"]["name"].lower() == "play":
                self._live = d["beatmap"]["time"]["live"]
            raw_mods = d["play"].get("mods")
            mod_list = raw_mods.get("array") if isinstance(raw_mods, dict) else raw_mods
            return {"source": "tosu", "path": path, "rate": rate, "cs": cs, "mods": mod_list,
                    "bid": d["beatmap"].get("id"), "client": d.get("client"),
                    "note": note if path else "selected map file not found" if mania else "not an osu!mania map"}
        except (KeyError, TypeError, AttributeError):
            return {"source": "tosu", "path": None, "rate": None,
                    "note": "unexpected tosu payload"}

    def _chart(self, d):
        """The selected chart's file, named by its SHA-256: lazer's store as is; stable's .osu linked into
        recdata.stable_store() (tosu under Wine can report a Songs folder that only exists inside Wine)."""
        rel = d["files"]["beatmap"]
        if not rel:
            return None
        if d.get("client") != "stable":
            root = next((r for r in (d["folders"].get("songs"), LAZER_FILES)
                         if r and os.path.isfile(os.path.join(r, rel))), None)
            return os.path.join(root, rel) if root else None
        if self._stable_songs is None:
            self._stable_songs = recdata.stable_songs(paths.stable_data())
        folder = d["folders"].get("beatmap") or ""
        real = next((p for p in (recdata.long_path(os.path.join(r, folder, rel))
                                 for r in (d["folders"].get("songs"), self._stable_songs) if r) if os.path.isfile(p)), None)
        if real is None:
            return None
        try:
            st = os.stat(real)
            key = (real, st.st_mtime, st.st_size)
            if key not in self._stable_files:
                with open(real, "rb") as fh:
                    sha = hashlib.sha256(fh.read()).hexdigest()
                if len(self._stable_files) > 256:
                    self._stable_files.clear()
                self._stable_files[key] = recdata.stable_link(real, sha)
            return self._stable_files[key]
        except OSError:
            return None

    def _track(self, state, d, path):
        """Attempt lifecycle: start on entering gameplay (a jump back in time is a restart), abort on
        leaving it anywhere but the results screen; the result shown next belongs to this attempt."""
        rel, tm = path and os.path.basename(path), d["beatmap"]["time"]
        mods = (d.get("play") or {}).get("mods")
        # Autoplay/Cinema is the game playing, not the player: no attempt
        auto = isinstance(mods, dict) and any(m.get("acronym") in ("AT", "CN") for m in mods.get("array") or [])
        if state == "play" and rel and not auto:
            live, first, last = tm.get("live") or 0, tm.get("firstObject") or 0, tm.get("lastObject") or 0
            p = self._play
            if p is None or p["sha"] != rel or live + 1500 < p["pos"]:
                # a jump back within the first seconds is gameplay taking over from the song-select preview
                # position, not a restart: the attempt starts again without an abort
                if p and (time.time() - p["t"] > 5 or p["sha"] != rel):
                    self.on_attempt("void" if self._disconnected else "abort", dict(p, end=time.time()))
                p = self._play = {"sha": rel, "path": path, "own": d.get("client") != "stable" or not self._replay_watching(),
                                  "rate": rate_from_mods(mods)[0], "t": time.time(), "pos": live, "progress": 0.0,
                                  "played_seconds": 0.,
                                  "mods": mods.get("array") if isinstance(mods, dict) else None}
                self.on_attempt("start", dict(p))
            delta = live - max(first, p["pos"])
            if 0 < delta < 2000 and p.get("rate"):
                p["played_seconds"] = p.get("played_seconds", 0.) + delta / 1000 / p["rate"]
            paused = bool((d.get("game") or {}).get("paused"))
            if paused and not p.get("paused_now"):
                p.setdefault("pauses", []).append(int(live))     # stable's .osr has no pause list (lazer's Realm does)
            p["paused_now"] = paused
            p["partial_stats"] = d.get("play", {}).get("hits")
            p["pos"], p["progress"] = live, max(0.0, min(1.0, (live - first) / max(1, last - first)))
        elif self._play:
            p, self._play = self._play, None
            if state == "resultscreen":
                self._done = p
            else:
                # auto: the first gameplay poll comes before Autoplay's mods do
                self.on_attempt("void" if self._disconnected or auto else "abort", dict(p, end=time.time()))
        self._disconnected = False

    @staticmethod
    def _replay_watching():
        """stable: tosu's StreamCompanion status (its v2 JSON lacks the flag) is 8 while a replay plays."""
        try:
            with urllib.request.urlopen(TOSU_URL.replace("/json/v2", "/json/sc"), timeout=0.6) as resp:
                return json.load(resp).get("status") == 8
        except (OSError, ValueError):
            return False

    def _note_score(self, d, path):
        """Hand the user's own mania result to the UI; fresh only right after its own gameplay (a replay or
        an old result being viewed is history, not today's form)."""
        rs, rel = d.get("resultsScreen") or {}, path and os.path.basename(path)
        name = rs.get("playerName")
        # tosu 4.26.2 reads stable's player name as "" or garbage ("wd" as guest, 2026-10-31): on stable the result
        # right after this app's own tracked, not-a-replay attempt on the chart is the player's
        own = name == (d.get("profile") or {}).get("name") or \
            d.get("client") == "stable" and bool(self._done and self._done["sha"] == rel and self._done.get("own"))
        if not rel or not own or str((rs.get("mode") or {}).get("name", "")).lower() != "mania":
            return
        score = {"sha256": rel, "rate": rate_from_mods(rs.get("mods"))[0], "client": d.get("client"),
                 "mods": (rs.get("mods") or {}).get("name", ""),
                 "accuracy": rs.get("accuracy"), "hits": rs.get("hits"),
                 "rank": rs.get("rank"), "max_combo": rs.get("maxCombo"),
                 "score": rs.get("score"), "played": rs.get("createdAt") or "",
                 "mods_list": (rs.get("mods") or {}).get("array")}
        if not sum(int(v or 0) for k, v in (score["hits"] or {}).items()
                   if k in ("geki", "300", "katu", "100", "50", "0")) or recdata.timestamp(score["played"]) is None:
            return                              # the results payload is still filling in
        if score != self._last_score:
            self._last_score = score
            att, fresh = self._done, False
            if att and att["sha"] == score["sha256"]:
                ts = recommend._ts(str(score["played"]))
                fresh = ts is None or ts >= att["t"] - 30
                if not fresh:
                    self.on_attempt("void", att)
            self._done = None
            self.on_score(score, path, fresh, att)

    # -- log fallback -------------------------------------------------------
    _lazer_running = staticmethod(recommend.lazer_running)

    def _poll_log(self):
        if recdata.client() == "stable":
            return {"source": None, "path": None, "rate": None, "note": "osu!stable needs tosu (starting it…)"}
        if not self._lazer_running():   # an old log is not a live selection
            self._log, self._log_display = (None, 0), None
            return {"source": None, "path": None, "rate": None,
                    "note": "osu!lazer is not running"}
        logs = glob.glob(os.path.join(LAZER_LOGS, "*runtime.log"))
        newest = max(logs, key=os.path.getmtime) if logs else None
        if newest and newest != self._log[0]:
            self._log = (newest, 0)
        if newest:
            try:
                with open(newest, "r", encoding="utf-8", errors="ignore") as fh:
                    fh.seek(self._log[1])
                    for line in fh:
                        m = self._LINE_RE.search(line)
                        if m:
                            self._log_display = m.group(1)
                    self._log = (newest, fh.tell())
            except OSError:
                pass
        path = self._resolve(self._log_display) if self._log_display else None
        note = "no live rate without tosu"
        if self._log_display and not path:
            note = f"not in library index: {self._log_display}"
        return {"source": "log", "path": path, "rate": None, "note": note}

    def _resolve(self, display):
        now = GLib.get_monotonic_time() / 1e6
        path = (self._index or {}).get(display)
        if (not path or not os.path.exists(path)) and now - self._last_reindex > 600 \
                and (self._index_job is None or not self._index_job.is_alive()):
            self._last_reindex = now
            force = self._index is not None
            self.on_status("tosu unavailable — indexing library in background; recovering scores from lazer")

            def scan():
                try:
                    self._index = lazer_index.load_or_build(store=LAZER_FILES, force=force)
                    if not force and display not in self._index:
                        self._index = lazer_index.load_or_build(store=LAZER_FILES, force=True)
                except Exception:
                    log_exc("library index")
            self._index_job = threading.Thread(target=scan, daemon=True, name="library-index")
            self._index_job.start()
        return path if path and os.path.exists(path) else None


# ---------------------------------------------------------------------------
def _map_markup(c):
    """One line: bold title, then small difficulty name and keymode."""
    meta = c.get("search_meta") or {}
    title, version = c.get("title", ""), meta.get("Version")
    if meta.get("Title"):
        title = " — ".join(x for x in (meta.get("Artist"), meta["Title"]) if x)
    elif " [" in title and title.rstrip().endswith("]"):
        title, version = title.split(" [", 1)
        version = version.rstrip()[:-1]
    esc = GLib.markup_escape_text
    return f"<b>{esc(title.removesuffix('.osu'))}</b> <small>{esc(f'[{version}] ') if version else ''}{c['keys']}K</small>"


class FixedBlock(Gtk.Box):
    """Stacked labels, each capped at n lines: longer text wraps, then ends in …, with the whole block as tooltip.
    The playlist's top is streamed and must never grow (user 2026-10-30); it may shrink, so no space is reserved
    ("no need to force the empty space")."""
    def __init__(self, *lines):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, no_show_all=True)
        self.labels = []
        for n in lines:
            label = Gtk.Label(xalign=0, wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR, max_width_chars=50,
                              ellipsize=Pango.EllipsizeMode.END, lines=n, no_show_all=True)
            self.labels.append(label)
            self.pack_start(label, False, False, 0)

    def set_parts(self, parts, tip=None):
        for label, markup in zip(self.labels, list(parts) + [""] * len(self.labels)):
            label.set_markup(markup or "")
            label.set_visible(bool(markup))
        self.set_visible(any(parts))
        self.set_tooltip_text(tip)

    def get_text(self):
        return "\n".join(label.get_text() for label in self.labels).rstrip("\n")


def target_markup(c, obs):
    """The picked map: [rate to set (green once osu! shows it) and expectation, title], so they never move."""
    if not c:
        return ["", ""]
    rate = c["rate"]
    wanted = "NM" if abs(rate-1.) < .0005 else "HT" if rate < 1. else "DT"
    actual = obs.get("rate")
    known = obs.get("source") == "tosu" and actual is not None
    sha = c.get("sha") or c.get("sha256")
    observed_sha = os.path.basename(obs.get("path") or "")
    same_chart = navigation.same_map(c, sha=observed_sha, bid=obs.get("bid"))
    mods = {m.get("acronym", "").upper() for m in (obs.get("mods") or [])}
    speed_mods = mods & {"DT", "NC", "HT", "DC", "WU", "WD", "AS"}
    mod_ok = (not speed_mods if wanted == "NM" else bool(speed_mods & ({"HT", "DC"} if wanted == "HT" else {"DT", "NC"})))
    correct = known and same_chart and abs(actual-rate) < .0005 and mod_ok
    colour = "#238747" if correct else "#d9384f" if known else "#8a9099"
    mark = "✓" if correct else "→"
    expected = f"~{100*recdata.shown_acc(c['acc_mid']):.1f}%" if c.get("acc_mid") is not None else "pending"
    if same_chart and sha and sha != observed_sha:
        expected = "Updated chart selected"
    return [f'<span foreground="{colour}" weight="bold">{mark} SET {rate:.2f}× · {wanted}</span>  <b>{expected}</b>',
            _map_markup(c)]


def card_markup(c, mode="pp", note=None):
    """The selected map: [numbers, title, details], numbers at a fixed place whatever the title (user 2026-10-30).
    PP gain leads the details line, so the numbers line never wraps. The full card is its tooltip."""
    if not c:
        return ["<small>" + GLib.markup_escape_text(note or "Select a map to see its prediction") + "</small>"], None
    m, sec = divmod(int(c["length"]), 60)
    skill = skill_calc.NAMES.get(c.get("skill"), c.get("skill") or "")
    line = (f"<small>SELECTED</small>  expected <b>~{100 * recdata.shown_acc(c['acc_mid']):.1f}%</b>"
            f" <small>({100 * recdata.shown_acc(c['acc_lo']):.1f}–{100 * recdata.shown_acc(c['acc_hi']):.1f}%)</small>  <b>{c.get('rate', 1.):.2f}×</b>")
    details = GLib.markup_escape_text(" · ".join(x for x in (f"{m}:{sec:02d}", skill, f"{c['nps']:.1f} NPS" if c.get("nps") else "") if x))
    if mode == "pp" and c["ranked"]:
        details = f"<b>+{c['gain']:.1f}pp</b> · {100 * c['p_up']:.0f}% to {'beat' if c['best'] else 'count'} · " + details
    parts = [line, _map_markup(c), f"<small>{details}</small>"]
    tip = [f"Expected ~{100 * recdata.shown_acc(c['acc_mid']):.1f}%, usually {100 * recdata.shown_acc(c['acc_lo']):.1f}–{100 * recdata.shown_acc(c['acc_hi']):.1f}%",
           f"{c.get('rate', 1.):.2f}× · {m}:{sec:02d} · {skill}"]
    if not c["ranked"]:
        tip.append("No pp (unranked or unranked rate)")
    else:
        tip.append(f"{c['pp_lo']:.0f}–{c['pp_hi']:.0f}pp · " + (f"best {c['best']:.0f} · " if c["best"] else "")
                   + f"+{c['gain']:.1f}pp expected · {100 * c['p_up']:.0f}% to {'beat' if c['best'] else 'count'}")
    if c.get("nps"):
        tip.append(f"{c['nps']:.1f} raw NPS")
    if c.get("plays_like"):
        stars, n = c["plays_like"]
        # User 2026-10-02 could not tell whose stars or whose scores ("7.4* osu SR? or our SR?").
        tip.append(f"Top-1000 players score on it like a {stars:.1f}★ map "
                   f"(ManiaScope ★, their best of {n:,} scores at {c.get('rate', 1.):.2f}×)")
    return parts, "\n".join(tip)


def result_markup(msg):
    """The last play: [rate · news, big accuracy vs expected, title]. No clock time: "00:44" read as the song's
    length (user 2026-10-31)."""
    esc = GLib.markup_escape_text
    head = " · ".join([f"{msg['rate']:.2f}×" if msg.get("rate") else "unknown rate", *msg.get("news", ())])
    acc = f"<big><b>{100 * msg['acc']:.2f}%</b></big>" if msg.get("acc") is not None else ""
    if msg.get("expected") is not None:
        acc += f" · expected ~{100 * msg['expected']:.2f}%"
    if msg.get("z") is not None:
        # Words, not σ (user 2026-10-31 preferred "better than expected").
        z = msg["z"]
        colour = "#238747" if z > .5 else "#d9384f" if z < -.5 else "#8a9099"
        acc += f' · <span foreground="{colour}" weight="bold">{recommend.verdict_words(z)}</span>'
    return [f"<small>LAST PLAY</small>  {esc(head)}", acc or esc(msg.get("text", "")), f"<b>{esc(msg['title'])}</b>"]


class ManiaScopeWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="ManiaScope")
        ui = self._load_ui()
        self._rec_mode = ui.get("recommendation_mode", "pp")
        if self._rec_mode not in recommend.MODES:
            self._rec_mode = "pp"
        self._rec_keys = {"pp": recdata.normalize_keys(ui.get("pp_keys")),
                          "nps": recdata.normalize_keys(ui.get("nps_keys") if ui.get("nps_keys") is not None else [7]),
                          "skills": recdata.normalize_keys(ui.get("skills_keys") if ui.get("skills_keys") is not None else [7])}
        # Never chosen: follow the keymodes actually played (7K until the lazer history is imported).
        self._keys_auto = {m for m in ("nps", "skills") if ui.get(f"{m}_keys") is None}
        self._key_checks, self._keys_syncing = {}, False
        self._practice_skills = skill_practice.normalize(ui.get("practice_skills"))
        self._skill_weights = dict(ui.get("skill_weights") or {})
        self._nps_focus = nps_playlist.normalize_focus(ui.get("nps_focus", .5))
        self._web_bias = {m: nps_playlist.normalize_focus((ui.get("web_bias") or {}).get(m, .5)) for m in ("nps", "skills")}
        self._acc_targets = {m: nps_playlist.normalize_target(ui.get(f"{m}_acc_target", nps_playlist.DEFAULT_TARGET))
                             for m in ("nps", "skills")}
        self._pp_settings = recommend.settled(recommend.PP_DEFAULTS, ui.get("pp_settings") or {}, None)
        self._pp_mix = recommend.settled(recommend.PP_MIX_DEFAULTS, ui.get("pp_mix") or {}, None)
        self._display = dict(DISPLAY_DEFAULTS, **{k: v for k, v in (ui.get("display") or {}).items()
                                                  if k in DISPLAY_DEFAULTS})
        self._next_action = ui.get("next_action", "auto")
        self._auto_next = ui.get("auto_next", True)
        self._keep_above = ui.get("keep_above", False)
        self._overlay = ui.get("overlay", False) and paths.WINDOWS
        if self._next_action not in ("auto", "copy", "link"):
            self._next_action = "auto"
        self._rec_revision, self._rec_views, self._card_data = 0, {}, None
        self._card_note = None
        self._selection_id = 0
        # `w`/`h` are the compact viewer geometry. Details may temporarily grow
        # the window, but collapsing must return to this size rather than letting
        # the timeline consume the space that the skill rows used.
        # Defaults: the user's own sizes on G835LX (2026-10-30).
        self._win_size = (int(ui.get("w", 442)), int(ui.get("h", 214)))
        self._details_size = (int(ui.get("details_w", 441)), int(ui.get("details_h", 663)))
        self._restoring_compact = False
        self._recs_size = (int(ui.get("recs_w", 442)), int(ui.get("recs_h", 420)))
        # Windows/X11 only: Wayland ignores move() (a compositor window rule places them there).
        self._recs_pos = tuple(ui["recs_pos"]) if ui.get("recs_pos") else None
        self._win_pos = tuple(ui["pos"]) if ui.get("pos") else None
        self.set_default_size(*(self._details_size if ui.get("details") else self._win_size))
        self._auto = ui.get("mode", "auto") == "auto"
        self._manual_rate = float(ui.get("rate", 1.0))
        self._obs = {"source": None, "path": None, "rate": None, "note": "starting…"}
        self._local = None          # pinned local file, or None to follow lazer
        self._shown = None          # (path, rate, chart, result, error, sv) on display
        self._pending = False       # a newer selection is being analysed (header shown, analysis cleared)
        self._gen = 0
        self._closed = False
        self._requested = None
        self._req = None            # latest (gen, path, rate); worker takes the newest
        self._req_cv = threading.Condition()
        self._save_timer = 0
        self._tosu_proc = None
        self._web_proc = self._web_win = None
        self._web_polling = False
        self._web_rows = []             # "Maps you don't have" rows waiting for the first website charts
        self._tosu_tried = 0.0
        self._live = None           # playback position (ms) while lazer is in gameplay
        self._context = ""
        self.tracking_db = recdata.connect()
        if self._keys_auto and recdata.played_keys(self.tracking_db):
            for m in self._keys_auto:
                self._rec_keys[m] = recdata.played_keys(self.tracking_db)
            self._keys_auto = set()
        elif self._keys_auto:
            GLib.timeout_add_seconds(15, self._auto_keys)

        bar = Gtk.HeaderBar(show_close_button=True)
        bar.set_custom_title(Gtk.Box())   # no title text: the controls need the width
        # Next, Skip and Tracking live on the playlist window's header (_recs_window): this one stays an OBS source.
        self.next_btn = Gtk.Button(label="Next · " + recommend.MODE_NAMES[self._rec_mode])
        self.next_btn.get_style_context().add_class("suggested-action")
        self.next_btn.set_tooltip_text("Pick from the active PP/NPS/Skills tab. Auto copies a local song-select search; set rate manually.")
        self.next_btn.connect("clicked", lambda _b: self._next(False))
        self.next_btn.set_sensitive(False)
        self.recs_btn = Gtk.ToggleButton(label="Playlist", always_show_image=True,
                                         image=Gtk.Image.new_from_icon_name("view-list-bullet-symbolic", Gtk.IconSize.BUTTON))
        self.recs_btn.set_tooltip_text("Show or hide the playlist window (Next, Skip, Tracking, PP/NPS/Skills lists)")
        bar.pack_start(self.recs_btn)
        skip = Gtk.Button(label="Skip", always_show_image=True,
                          image=Gtk.Image.new_from_icon_name("media-skip-forward-symbolic", Gtk.IconSize.BUTTON))
        skip.set_tooltip_text("Skip: not this map (avoided for about 12 h)")
        skip.connect("clicked", lambda _b: self._next(True))
        self.skip_btn = skip
        skip.set_sensitive(False)
        self.track_btn = None
        if tracking_switch(self.tracking_db):
            self.track_btn = Gtk.ToggleButton(label="Tracking on", active=recdata.tracking_enabled(self.tracking_db))
            self.track_btn.set_label("Tracking on" if self.track_btn.get_active() else "Tracking off")
            self.track_btn.set_tooltip_text("Paused plays stay excluded after imports and restarts. Resume for the next play.")
            self.track_btn.connect("toggled", self._tracking_toggled)
        bar.pack_end(self._overflow())
        self.set_titlebar(bar)
        self.set_title("ManiaScope")   # taskbar name; the header bar shows none
        if self._win_pos:              # after the titlebar: it changes the frame move() is relative to
            self.move(*self._win_pos)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        for side in ("start", "end", "top"):
            getattr(root, f"set_margin_{side}")(12 if side != "top" else 8)
        self.add(root)

        def label(dim=False, **kw):
            lb = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END, **kw)
            if dim:
                lb.get_style_context().add_class("dim-label")
            return lb

        self.title_lbl = label()
        self.version_lbl = label(dim=True)
        self.ln_pct_lbl = Gtk.Label(xalign=1)
        self.ln_pct_lbl.get_style_context().add_class("dim-label")
        self.ln_pct_lbl.set_tooltip_text("Long-note heads / all note heads. Each LN counts once; tails are not extra notes.")
        root.pack_start(self.title_lbl, False, False, 0)
        metadata = Gtk.Box(spacing=8)
        metadata.pack_start(self.version_lbl, True, True, 0)
        metadata.pack_end(self.ln_pct_lbl, False, False, 0)
        root.pack_start(metadata, False, False, 0)

        # context (keys, dan, rate, length) on its own full-width line from the window's left edge
        self.context_lbl = label(dim=True, hexpand=True)
        root.pack_start(self.context_lbl, False, False, 0)
        head = Gtk.Box(spacing=8)
        self.number_lbl = GradLabel(valign=Gtk.Align.CENTER)
        self.number_lbl.set_tooltip_text(
            "Overall difficulty: level a well-rounded player needs for controlled, "
            "accurate play. Provisional calibration — one decimal is already generous.")
        head.pack_start(self.number_lbl, False, False, 0)
        side = Gtk.Grid(valign=Gtk.Align.CENTER, column_spacing=6)
        self.dominant_lbl = label(hexpand=True)
        # every line always present (blank when empty) so the timeline keeps one height
        self.others_lbl = label()
        self.tags_lbl = label(dim=True)
        # NPS beside the top skill; lower lines use the full width
        self.nps_lbl = GradLabel(valign=Gtk.Align.CENTER)
        self.nps_lbl.set_tooltip_text("Notes per second, first to last note, at this rate")
        side.attach(self.dominant_lbl, 0, 0, 1, 1)
        side.attach(self.nps_lbl, 1, 0, 1, 1)
        side.attach(self.others_lbl, 0, 1, 2, 1)
        side.attach(self.tags_lbl, 0, 2, 2, 1)
        head.pack_start(side, True, True, 0)
        self.details_btn = Gtk.ToggleButton(valign=Gtk.Align.START, relief=Gtk.ReliefStyle.NONE)
        self.details_btn.add(Gtk.Image.new_from_icon_name("pan-down-symbolic", Gtk.IconSize.BUTTON))
        self.details_btn.set_tooltip_text("Skill details")
        head.pack_end(self.details_btn, False, False, 0)
        root.pack_start(head, False, False, 0)

        self.timeline = Gtk.DrawingArea()
        self.timeline.set_size_request(-1, 30)
        self.timeline.set_tooltip_text("Difficulty over time (2 s horizon of the rating's own "
                                       "demand curve, scaled to this map's peak). Fills while you play.")
        self.timeline.connect("draw", self.on_draw_timeline)
        root.pack_start(self.timeline, True, True, 2)
        self._recs_window()
        self.set_keep_above(self._keep_above)
        self.recs.set_keep_above(self._keep_above)
        if ui.get("recs_open"):           # open last time the app closed: reopen once the main window is up
            GLib.idle_add(lambda: self.recs_btn.set_active(True))

        self.details = Gtk.Revealer(transition_type=Gtk.RevealerTransitionType.NONE)
        grid = self._grid = Gtk.Grid(column_spacing=8, row_spacing=1)
        self._bars = {}
        for i, key in enumerate(skill_calc.NAMES):
            name = Gtk.Label(xalign=0)
            pb = Gtk.ProgressBar(hexpand=True, valign=Gtk.Align.CENTER)
            val = Gtk.Label(xalign=1, width_chars=5)
            grid.attach(name, 0, i, 1, 1)
            grid.attach(pb, 1, i, 1, 1)
            grid.attach(val, 2, i, 1, 1)
            self._bars[key] = (pb, val, name)
        grid.set_tooltip_text(
            "Each skill is the rating of the sections that show it, on the same scale as the "
            "overall number: a section can count toward several skills (Delay and Technical, "
            "Chordstream and Bracket); overall charges its work once. Bars: fixed "
            f"0–{BAR_FULL:g}, comparable across maps. Stamina is the hardest 4 minutes, "
            "a shorter map counted with rest.")
        self.details.add(grid)
        root.pack_start(self.details, False, False, 0)
        self.details_btn.connect("toggled", self.on_details_toggled)
        self.details_btn.set_active(bool(ui.get("details")))

        foot = Gtk.Box(spacing=6)
        self.status = label(dim=True)
        self.status.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.85"))
        foot.pack_start(self.status, True, True, 0)
        self.follow_btn = Gtk.Button(label="Follow osu!", no_show_all=True,
                                     relief=Gtk.ReliefStyle.NONE)
        self.follow_btn.connect("clicked", self.on_follow)
        foot.pack_end(self.follow_btn, False, False, 0)
        foot.set_margin_bottom(4)
        root.pack_start(foot, False, False, 0)

        self._apply_display()
        self._render()
        threading.Thread(target=self._worker, daemon=True).start()
        self._start_tosu()
        self.rec = recommend.Worker(lambda m: GLib.idle_add(self._on_rec, m), busy=lambda: self._live is not None,
                                    game=lambda: self._obs.get("source") == "tosu" and bool(self._obs.get("path")))
        self._sync_rec()
        self.rec.start()
        self.watcher = LazerWatcher(lambda o: GLib.idle_add(self._on_obs, o),
                                    lambda s: GLib.idle_add(self._on_status, s),
                                    lambda t: GLib.idle_add(self._on_progress, t),
                                    lambda *a: GLib.idle_add(self._on_score, *a),
                                    self.rec.put)
        self.watcher.start()
        self._tosu_timer = GLib.timeout_add_seconds(5, self._check_tosu)
        # GTK/Wayland may not emit the X11 configure events used previously.
        # Allocation works on both backends and also covers the playlist window.
        self.connect("size-allocate", self._on_configure)
        self.connect("configure-event", self._on_moved)
        self.connect("destroy", self._on_destroy)
        self.connect("delete-event", lambda *_: (lifecycle("close requested"), False)[1])
        self.connect("unmap-event", lambda *_: (lifecycle("window unmapped"), False)[1])
        if first_run_prompt(self.tracking_db):
            GLib.timeout_add(800, lambda: self._profile_dialog(first=True) and False)

    def _overflow(self):
        """⋯: the viewer's own controls (Auto, rate, Open, Save, Saved maps) and the few session settings.

        A Gtk.Menu, not a popover: on Wayland a popover is a subsurface the compositor never moves, so a
        tall one flipped above a window at the screen top was cut off (user 2026-10-01); a menu is an
        xdg_popup that the compositor flips and slides onto the screen."""
        menu = Gtk.Menu()

        def item(text, cb, tip=None, into=menu):
            mi = Gtk.MenuItem(label=text)
            mi.set_tooltip_text(tip)
            mi.connect("activate", cb)
            into.append(mi)
            return mi

        def choices(title, options, active, cb, tip=None):
            sub, group, items = Gtk.Menu(), None, {}
            for value, text in options:
                mi = Gtk.RadioMenuItem.new_with_label_from_widget(group, text)
                group = mi
                mi.set_active(value == active)
                mi.connect("toggled", lambda m, v=value: m.get_active() and cb(v))
                sub.append(mi)
                items[value] = mi
            parent = Gtk.MenuItem(label=title)
            parent.set_tooltip_text(tip)
            parent.set_submenu(sub)
            menu.append(parent)
            return items

        self.auto_btn = Gtk.CheckMenuItem(label="Follow osu!'s rate (Auto)", active=self._auto)
        self._auto_handler = self.auto_btn.connect("toggled", self.on_auto_toggled)
        menu.append(self.auto_btn)
        adj = Gtk.Adjustment(value=self._manual_rate, lower=0.25, upper=3.0,
                             step_increment=0.01, page_increment=0.05)
        self.rate_spin = Gtk.SpinButton(adjustment=adj, digits=2, numeric=True)
        self.rate_spin.set_tooltip_text("Playback rate — editing switches to manual")
        self._spin_handler = self.rate_spin.connect("value-changed", self.on_rate_edited)
        item("Rate…", lambda _m: self._rate_dialog(), "Set the playback rate by hand (switches Auto off)")
        if DEBUG:
            item("Open .osu…", self.on_open)
            item("Save this map at this rate ★", self.on_save_map, "Saved maps can be compared with each other")
            item("Saved maps & comparisons…", self.on_compare)
        item("Display…", lambda _m: self._display_dialog(), "Choose what the main window shows")
        menu.append(Gtk.SeparatorMenuItem())
        item("Skip warmup", lambda _m: self.rec.put("skip_warmup"), "Already warm: remove the cold-session prior for PP, NPS and Skills")
        item("New session", lambda _m: self.rec.put("reset"), "Forget today's form (history and bests stay)")
        self._mode_items = choices("Next playlist", (("pp", "PP opportunities"), ("nps", "NPS playlist"),
                                                     ("skills", "Skill practice")),
                                   self._rec_mode, self._set_rec_mode)
        choices("Next action", (("auto", "Auto"), ("copy", "Copy search"), ("link", "Open beatmap link")),
                self._next_action, self._set_next_action,
                "Auto copies search for installed maps and opens links for missing PP maps. Local-only maps always use search. Nothing sets mods or starts play.")
        auto_next = Gtk.CheckMenuItem(label="Auto next after finishing the pick", active=self._auto_next)
        auto_next.set_tooltip_text("Finishing the recommended map picks the next one and copies its search, as if you pressed Next")
        auto_next.connect("toggled", lambda m: (setattr(self, "_auto_next", m.get_active()), self._queue_save()))
        menu.append(auto_next)
        above = Gtk.CheckMenuItem(label="Always on top", active=self._keep_above)
        above.set_tooltip_text("Keep ManiaScope and the playlist above other windows. Not above exclusive fullscreen; "
                               "for that use the in-game overlay.")
        above.connect("toggled", lambda m: self._set_keep_above(m.get_active()))
        menu.append(above)
        if paths.WINDOWS:
            overlay = Gtk.CheckMenuItem(label="In-game overlay (tosu)", active=self._overlay)
            overlay.set_tooltip_text("Show this window inside osu!, also in fullscreen, through tosu's in-game overlay")
            overlay.connect("toggled", self._toggle_overlay)
            menu.append(overlay)
        item("Reset learned NPS taste", lambda _m: self.rec.put("reset_taste"), "Keep scores, session and map history; forget only learned taste")
        item("Import osu! scores now", lambda _m: self.rec.put("import"))
        if not os.access(TOSU_BIN, os.X_OK):
            item("Download tosu (rate tracking)", lambda _m: self._install_tosu(),
                 "Follows the exact rate you pick in osu!. Official tosu release from GitHub (~40 MB).")
        item("osu! profile…", lambda _m: self._profile_dialog(),
             "Your username: your public top 100 plays fill in bests that lazer doesn't have locally")
        if DEBUG:
            item("Play & prediction history…", self._show_history)
        menu.show_all()
        button = Gtk.MenuButton(popup=menu)
        button.add(Gtk.Image.new_from_icon_name("open-menu-symbolic", Gtk.IconSize.BUTTON))
        button.set_tooltip_text("More")
        return button

    def _small_dialog(self, title, child):
        """A small non-modal window the compositor places: stays on screen wherever the main window sits."""
        dlg = Gtk.Dialog(title=title, transient_for=self, modal=False, destroy_with_parent=True)
        dlg.add_button("Close", Gtk.ResponseType.CLOSE)
        dlg.connect("response", lambda d, _r: d.destroy())
        area = dlg.get_content_area()
        area.set_margin_start(10); area.set_margin_end(10); area.set_margin_top(8)
        area.pack_start(child, False, False, 0)
        dlg.show_all()
        return dlg

    def _rate_dialog(self):
        row = Gtk.Box(spacing=6)
        row.pack_start(Gtk.Label(label="Rate"), False, False, 0)
        row.pack_start(self.rate_spin, True, True, 0)
        # the spin outlives the dialog: _set_spin keeps it in step with lazer while it is closed
        self._small_dialog("Rate", row).connect("destroy", lambda _d: row.remove(self.rate_spin))

    def _profile_dialog(self, first=False):
        dlg = Gtk.Dialog(title="Welcome to ManiaScope" if first else "osu! profile", parent=self, modal=True)
        dlg.add_buttons("Skip" if first else "Cancel", Gtk.ResponseType.CANCEL, "Save", Gtk.ResponseType.OK)
        box = dlg.get_content_area()
        box.set_spacing(8)
        box.set_margin_start(12); box.set_margin_end(12); box.set_margin_top(12)
        text = Gtk.Label(label="Your osu! username or profile number. ManiaScope reads your public top 100 mania "
                               "plays (no password or API key) so PP suggestions know your bests. Empty = off.",
                         wrap=True, max_width_chars=48, xalign=0)
        entry = Gtk.Entry(activates_default=True)
        entry.set_text(recdata.kv_get(self.tracking_db, "website_username") or "")
        dlg.set_default_response(Gtk.ResponseType.OK)
        box.pack_start(text, False, False, 0)
        box.pack_start(entry, False, False, 0)
        dlg.show_all()
        if dlg.run() == Gtk.ResponseType.OK:
            self.rec.put("website_user", entry.get_text())
        dlg.destroy()

    def _display_dialog(self):
        """⋯ → Display…: what the main window shows. Persisted with the window layout."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        for key, text in (("rating", "Difficulty rating"), ("dan", "Dan level"), ("nps", "NPS"),
                          ("ln", "LN %"), ("timeline", "Difficulty timeline")):
            check = Gtk.CheckButton(label=text, active=self._display[key])
            check.connect("toggled", lambda b, k=key: self._set_display(k, b.get_active()))
            box.pack_start(check, False, False, 0)
        row = Gtk.Box(spacing=6)
        row.pack_start(Gtk.Label(label="Skills shown"), False, False, 0)
        spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(value=self._display["skills"], lower=1, upper=6,
                                                         step_increment=1), digits=0, numeric=True)
        spin.set_tooltip_text(f"Default {DISPLAY_DEFAULTS['skills']}")
        spin.connect("value-changed", lambda s: self._set_display("skills", int(s.get_value())))
        row.pack_start(spin, False, False, 0)
        box.pack_start(row, False, False, 0)
        return self._small_dialog("Display", box)

    def _set_display(self, key, value):
        if self._display.get(key) != value:
            self._display[key] = value
            self._apply_display()
            self._render()
            self._queue_save()

    def _apply_display(self):
        for key, widget in (("rating", self.number_lbl), ("nps", self.nps_lbl),
                            ("ln", self.ln_pct_lbl), ("timeline", self.timeline)):
            widget.set_no_show_all(True)
            widget.set_visible(bool(self._display[key]))

    def _recommended_stars(self, keys):
        """osu! ★ range (1.0×) covering the player's recent plays in these keymodes at 0.9–1.2×, or None."""
        try:
            rows = self.tracking_db.execute(
                "SELECT info FROM events WHERE kind='start' ORDER BY id DESC LIMIT 400").fetchall()
        except Exception:
            return None
        stars = sorted(f["stars"] for f in ((json.loads(r[0]).get("features") or {}) for r in rows)
                       if f.get("stars") and f.get("keys") in keys)
        if len(stars) < 30:
            # A new install has no tracked starts yet: its imported lazer plays say the same thing.
            calc = recdata.calc_id()
            plays = self.tracking_db.execute(
                "SELECT f.data FROM scores s JOIN feats f ON f.key = s.sha256 || '@' || printf('%.3f', s.rate) || '|' || ? "
                "WHERE s.src != 'public' AND s.rate BETWEEN .9 AND 1.2 ORDER BY s.played DESC LIMIT 400", (calc,))
            stars = sorted(f["stars"] for f in (recdata.feats_load(r[0]) for r in plays) if f.get("stars") and f.get("keys") in keys)
        if len(stars) < 30:
            return None
        lo, hi = stars[len(stars) // 10], stars[len(stars) * 9 // 10]
        return round(lo / 1.2 * 2) / 2, round(hi / .9 * 2) / 2

    def _web_charts_row(self):
        """Settings → its own window, so size, time and progress are visible before and during a download."""
        self.web_btn = Gtk.Button(label="Website charts…")
        self.web_btn.set_tooltip_text("Add maps you don't have to the playlists (chart text only, no audio).")
        self.web_btn.connect("clicked", lambda _b: self._web_window())
        return self.web_btn

    def _web_run(self, command, keys, stars=None):
        argv = ([sys.executable, "--webmaps"] if getattr(sys, "frozen", False)
                else [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "webmaps.py")])
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._web_proc = subprocess.Popen(
            argv + [command, "--keys", ",".join(map(str, keys))] + (["--stars", "%g-%g" % stars] if stars else []),
            stdout=open(os.path.join(CACHE_DIR, "webmaps.log"), "w"), stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._web_poll()

    def _web_window(self):
        import webmaps
        if self._web_win:
            self._web_win.present()
            return
        win = self._web_win = Gtk.Window(title="ManiaScope — Website charts", transient_for=self, default_width=480)
        win.connect("destroy", lambda _w: setattr(self, "_web_win", None))
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=14)
        win.add(box)
        intro = Gtk.Label(xalign=0, wrap=True, max_width_chars=60, label=(
            "Downloads the chart text (no audio) of maps you don't have from osu.direct, analyses it and adds "
            "those maps to the NPS and Skills playlists; Next then offers the download in osu!. Runs at low "
            "priority. Pause any time; closing ManiaScope pauses it and Resume continues where it stopped."))
        box.pack_start(intro, False, False, 0)
        try:
            db = webmaps._db()
            saved, saved_keys = webmaps._kv(db, "stars"), webmaps._kv(db, "want_keys")
            db.close()
        except Exception:
            saved = saved_keys = None
        keys_row = Gtk.Box(spacing=6)
        keys_row.pack_start(Gtk.Label(label="Keymodes"), False, False, 0)
        checks = {}
        for k in recdata.SUPPORTED_KEYS:
            checks[k] = Gtk.CheckButton(label=f"{k}K", active=k in (saved_keys or set(self._rec_keys["nps"]) |
                                                                     set(self._rec_keys["skills"])))
            keys_row.pack_start(checks[k], False, False, 0)
        box.pack_start(keys_row, False, False, 0)
        row = Gtk.Box(spacing=6)
        row.pack_start(Gtk.Label(label="osu! ★ at 1.0×"), False, False, 0)
        selected = lambda: tuple(k for k, b in checks.items() if b.get_active())
        recommended = self._recommended_stars(selected())
        spins = []
        for value in saved or recommended or (5., 10.):
            spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(value=value, lower=0, upper=15, step_increment=.5),
                                  digits=1, numeric=True)
            spin.connect("value-changed", lambda _s: self._web_poll())
            spins.append(spin)
            row.pack_start(spin, False, False, 0)
        hint = Gtk.Label()
        row.pack_start(hint, False, False, 0)
        box.pack_start(row, False, False, 0)

        def keys_changed(b=None):
            r = self._recommended_stars(selected())
            hint.set_text(f"(your plays suggest {r[0]:g}–{r[1]:g})" if r else "")
            if b:
                self._web_poll()
        for b in checks.values():
            b.connect("toggled", keys_changed)
        keys_changed()
        summary = Gtk.Label(xalign=0, wrap=True, selectable=True, max_width_chars=60,
                            label="Counting the charts in this range…")
        box.pack_start(summary, False, False, 0)
        bar = Gtk.ProgressBar(show_text=True, no_show_all=True)
        box.pack_start(bar, False, False, 0)
        buttons = Gtk.Box(spacing=6)
        check = Gtk.Button(label="Check for new charts")
        go = Gtk.Button(label="Start", sensitive=False)
        buttons.pack_end(go, False, False, 0)
        buttons.pack_end(check, False, False, 0)
        box.pack_start(buttons, False, False, 0)
        self._web_view = dict(spins=spins, checks=checks, selected=selected, summary=summary, bar=bar, check=check, go=go, state=None)
        check.connect("clicked", lambda _b: self._web_run("check", selected()))

        def clicked(_b):
            if self._web_view["state"] == "running":
                open(os.path.join(webmaps.WEB_DIR, "stop"), "w").close()      # pauses after the current chart
            else:
                self._web_run("crawl", selected(), tuple(sorted(s.get_value() for s in spins)))
        go.connect("clicked", clicked)
        win.show_all()
        self._web_poll()

    def _web_poll(self):
        """Read the crawl's progress off the UI thread; repeats while a crawl runs or the window is open."""
        if self._web_polling:
            return
        self._web_polling = True
        view = self._web_win and self._web_view
        stars = tuple(sorted(s.get_value() for s in view["spins"])) if view else None
        keys = set(view["selected"]()) if view else set()

        def work():
            import webmaps
            try:
                db = webmaps._db()
                run = webmaps.running()
                info = dict(progress=webmaps._kv(db, "progress") or {}, running=run, disk=webmaps.disk_bytes(),
                            estimate=webmaps.estimate(db, keys, stars) if view and not run else None)
                p = info["progress"]
                if info["running"] and p.get("fetch_total") is not None:
                    info["left"] = webmaps.seconds(db, p["fetch_total"] - p.get("fetched", 0),
                                                   p["analyse_total"] - p.get("analysed", 0))
                    info["done"] = 1 - info["left"] / max(1., webmaps.seconds(db, p["fetch_total"], p["analyse_total"]))
                db.close()
            except Exception as exc:
                info = dict(error=f"{type(exc).__name__}: {exc}")
            GLib.idle_add(self._web_show, info)
        threading.Thread(target=work, daemon=True).start()

    def _web_show(self, info):
        self._web_polling = False
        if self._closed:
            return False
        p, est = info.get("progress", {}), info.get("estimate")
        run = info.get("running")
        if p.get("ready", 0) != getattr(self, "_web_ready", None):
            if getattr(self, "_web_ready", None) is not None:
                self.rec.put("web_updated")      # new website charts join the open playlist without a restart
            self._web_ready = p.get("ready", 0)
            for row in self._web_rows if self._web_ready else ():
                row.set_sensitive(True)
        done = max(0., info.get("done", 0.))
        self.web_btn.set_label(f"Website charts… {100 * done:.0f}%" if run and p.get("phase") != "check"
                               else "Website charts… paused" if p.get("phase") in ("paused", "failed") else "Website charts…")
        view = self._web_view if self._web_win else None
        if view:
            summary, bar, go, check = view["summary"], view["bar"], view["go"], view["check"]
            if "error" in info:
                summary.set_text("Could not read the website chart state: " + info["error"])
            elif run and p.get("phase") == "check":
                summary.set_text(f"Checking osu.direct: {p.get('sets', 0):,} map sets listed so far "
                                 "(the first check reads the whole mania listing, about 15–30 min).")
                bar.pulse()
                bar.set_text("checking")
            elif run:
                summary.set_text(
                    f"Downloaded {p.get('fetched', 0):,} of {p.get('fetch_total', 0):,}, analysed "
                    f"{p.get('analysed', 0):,} of {p.get('analyse_total', 0):,}. "
                    f"{p.get('ready', 0):,} charts are in the playlist already "
                    f"(the first after 300, then in batches).\nAbout {_duration(info.get('left', 0))} left · "
                    f"{_size(info['disk'])} on disk now.")
                bar.set_fraction(min(1., done))
                bar.set_text(f"{100 * done:.0f}%")
            else:
                lines = []
                if p.get("phase") == "failed":
                    lines.append(f"Stopped with an error: {p.get('error')}. Resume retries.")
                elif p.get("phase") == "paused":
                    lines.append("Paused.")
                if est["unlisted"]:
                    lines.append(f"{recdata.keys_label(est['unlisted'])} not listed yet. Start first reads osu.direct's "
                                 "mania listing (about 15–30 min the first time, seconds after that), then downloads; "
                                 "Check only reads the listing, to see the size first.")
                if not view["selected"]():
                    lines.append("Pick at least one keymode.")
                elif est["fetch"] or est["analyse"]:
                    lines.append(f"To do: {est['fetch']:,} charts to download (about {_size(est['new_bytes'])}) "
                                 f"and {est['analyse']:,} to analyse, about {_duration(est['seconds'])} on this PC.")
                    lines.append(f"On disk now {_size(est['disk'])}, afterwards about "
                                 f"{_size(est['disk'] + est['new_bytes'])}. {est['ready']:,} charts in this range "
                                 "are in the playlist already.")
                elif not est["unlisted"]:
                    lines.append(f"Up to date: {est['ready']:,} charts in this range are in the playlist, "
                                 f"{_size(est['disk'])} on disk.")
                if est["checked"]:
                    lines.append(f"Listing checked {_duration(time.time() - est['checked'])} ago.")
                summary.set_text("\n".join(lines))
            bar.set_visible(bool(run))
            for w in view["spins"] + list(view["checks"].values()):
                w.set_sensitive(not run)
            view["state"] = "running" if run else None
            go.set_label("Pause" if run else "Resume" if p.get("phase") in ("download", "analyse", "paused", "failed")
                         else "Start")
            # Start lists unchecked keymodes itself: a first run needs no separate Check (fresh install).
            go.set_sensitive(bool(run or (est and (est["fetch"] or est["analyse"] or est["unlisted"]))))
            check.set_sensitive(not run and bool(view["selected"]()))
        proc = self._web_proc
        if view or run or (proc and proc.poll() is None):
            GLib.timeout_add_seconds(2, lambda: (self._web_poll(), False)[1])
        return False

    @staticmethod
    def _slider(label, value, show, changed, lo=0., hi=1., step=.05, values=None, tooltip=None):
        """A settings row: name, slider, current value. values: a list the slider steps through instead.
        changed(v) runs on user moves only."""
        row = Gtk.Box(spacing=8)
        row.pack_start(Gtk.Label(label=label, xalign=0, width_chars=16), False, False, 0)
        text = Gtk.Label(width_chars=11, xalign=0)
        if values:
            lo, hi, step = 0, len(values) - 1, 1
            value = min(range(len(values)), key=lambda i: abs(values[i] - value))
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, lo, hi, step)
        scale.set_draw_value(False)
        scale.set_value(value)
        scale.get_accessible().set_name(label)
        row.set_tooltip_text(tooltip)
        def moved(s, initial=False):
            v = values[int(round(s.get_value()))] if values else round(round(s.get_value() / step) * step, 6)
            text.set_text(show(v))
            if not initial:
                changed(v)
        scale.connect("value-changed", moved)
        moved(scale, True)
        row.pack_start(scale, True, True, 0)
        row.pack_end(text, False, False, 0)
        return row

    @staticmethod
    def _switch(label, active, changed, tooltip=None):
        row = Gtk.Box(spacing=8)
        row.pack_start(Gtk.Label(label=label, xalign=0), False, False, 0)
        switch = Gtk.Switch(active=active, valign=Gtk.Align.CENTER)
        switch.get_accessible().set_name(label)
        switch.connect("notify::active", lambda w, _p: changed(w.get_active()))
        row.set_tooltip_text(tooltip)
        row.pack_end(switch, False, False, 0)
        return row

    @staticmethod
    def _section(box, title):
        head = Gtk.Label(xalign=0, margin_top=8)
        head.set_markup(f"<small><b>{title}</b></small>")
        head.get_style_context().add_class("dim-label")
        box.pack_start(head, False, False, 0)

    def _web_bias_row(self, mode):
        """How often this tab suggests website charts (maps you don't have) next to your own."""
        def show(v):
            f = nps_playlist.web_factor(v)
            return f"{f / (1 + f):.0%} of picks" if f else "never"
        def changed(v):
            if v != self._web_bias[mode]:
                self._web_bias[mode] = v
                self._sync_rec()
        row = self._slider("Maps you don't have", self._web_bias[mode], show, changed,
                           tooltip="Website charts (Website charts…): their share of your picks. Middle: half. "
                                   "Right end: 80 %. Left end: never, only maps you own.")
        # Without website charts it can change nothing (Less, 2026-10-31: set "Never", saw no difference).
        if not os.path.exists(os.path.join(recdata.REC_DIR, "web", "web.pkl")):
            row.set_sensitive(False)
            row.set_tooltip_text("Nothing to mix yet: add charts with Website charts… on the NPS tab. Until then "
                                 "every pick is a map you own.")
            self._web_rows.append(row)
        return row

    def _acc_target_row(self, mode):
        """The displayed-accuracy target this tab's picks aim at."""
        def changed(v):
            target = nps_playlist.normalize_target(v / 100)
            if target != self._acc_targets[mode]:
                self._acc_targets[mode] = target
                self._sync_rec()
        return self._slider("Accuracy target", 100 * self._acc_targets[mode],
                            lambda v: f"{v:.1f}%" + (" (default)" if v / 100 == nps_playlist.DEFAULT_TARGET else ""),
                            changed, 88., 98., .5, tooltip="Maps and rates are chosen so the predicted accuracy lands near this.")

    def _set_option(self, store, key, value):
        if store[key] != value:
            store[key] = value
            self._sync_rec()

    def _revealed(self, box, active, rows):
        """Rows that only show while their switch is on."""
        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        for row in rows:
            inner.pack_start(row, False, False, 0)
        revealer = Gtk.Revealer(reveal_child=active)
        revealer.add(inner)
        box.pack_start(revealer, False, False, 0)
        return revealer

    def _settings_page(self, mode, key_btn):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=6)
        keys = Gtk.Box(spacing=8)
        keys.pack_start(Gtk.Label(label="Keymodes", xalign=0, width_chars=16), False, False, 0)
        keys.pack_start(key_btn, True, True, 0)
        pct = lambda v: f"{100 * v:.0f}%"
        if mode == "nps":
            self._section(box, "MAPS")
            box.pack_start(keys, False, False, 0)
            box.pack_start(self._acc_target_row(mode), False, False, 0)
            self._section(box, "MIX")
            box.pack_start(self._slider("Variety to high NPS", self._nps_focus, lambda v: f"{100 * v:.0f}% NPS",
                                        self._set_nps_focus, tooltip="Left: pattern types (jacks, streams, LN…) get an "
                                        "even share of the list. Right: the highest NPS maps, mostly chordstreams and "
                                        "brackets. Both aim at the accuracy target."),
                           False, False, 0)
            box.pack_start(self._web_bias_row(mode), False, False, 0)
            web = Gtk.Box()
            web.pack_end(self._web_charts_row(), False, False, 0)
            box.pack_start(web, False, False, 0)
            GLib.idle_add(self._web_poll)      # a paused download shows on the button after a restart
            self._section(box, "PP MAPS IN THIS LIST")
            mix = self._pp_mix
            rows = [self._slider("One PP map every", mix["every"], lambda v: f"{v:.0f} picks",
                                 lambda v: self._set_option(mix, "every", int(v)), 2, 8, 1),
                    self._slider("Min gain", mix["min_gain"], lambda v: f"{v:g}pp",
                                 lambda v: self._set_option(mix, "min_gain", v), values=(1., 2., 3., 5., 8., 10., 15.)),
                    self._slider("Min chance", mix["chance"], pct, lambda v: self._set_option(mix, "chance", v),
                                 .2, .9, .05, tooltip="Chance to beat your best, NM and DT. It already includes "
                                                      "this session's form."),
                    self._slider("Min chance (HT)", mix["ht_chance"], pct,
                                 lambda v: self._set_option(mix, "ht_chance", v), .4, .95, .05)]
            def mix_on(on):
                revealer.set_reveal_child(on)
                self._set_option(mix, "on", on)
            box.pack_start(self._switch("Mix in likely PP gains", mix["on"], mix_on,
                                        "PP maps you are likely to beat now, between NPS maps. Off: no PP "
                                        "scoring while you play NPS."), False, False, 0)
            revealer = self._revealed(box, mix["on"], rows)
        elif mode == "skills":
            self._section(box, "SKILLS")
            self.practice_btn = self._skill_picker()
            box.pack_start(self.practice_btn, False, False, 0)
            self._section(box, "MAPS")
            box.pack_start(keys, False, False, 0)
            box.pack_start(self._acc_target_row(mode), False, False, 0)
            box.pack_start(self._web_bias_row(mode), False, False, 0)
        else:
            opt = self._pp_settings
            set_ = lambda key: lambda v: self._set_option(opt, key, v)
            self._section(box, "CHOICE")
            box.pack_start(keys, False, False, 0)
            box.pack_start(self._slider("Min gain", opt["min_gain"], lambda v: f"{v:g}pp", set_("min_gain"),
                                        values=(.01, .1, .25, .5, 1., 2., 3., 5., 10.),
                                        tooltip="Expected account pp gain a map needs."), False, False, 0)
            box.pack_start(self._slider("Min chance", opt["min_chance"], pct, set_("min_chance"), .05, .6, .01,
                                        tooltip="Chance to beat your best. Maps above your usual level keep at "
                                                "least 25%."), False, False, 0)
            box.pack_start(self._switch("Top-100 maps only", opt["top100"], set_("top100"),
                                        "Only maps that are, or would enter, your top 100."), False, False, 0)
            self._section(box, "MODS")
            box.pack_start(self._switch("DT", opt["dt"], set_("dt")), False, False, 0)
            def ht_on(on):
                revealer.set_reveal_child(on)
                set_("ht")(on)
            box.pack_start(self._switch("HT", opt["ht"], ht_on, "HT only as a likely, real improvement."),
                           False, False, 0)
            revealer = self._revealed(box, opt["ht"], [
                self._slider("HT min chance", opt["ht_chance"], pct, set_("ht_chance"), .3, .95, .05),
                self._slider("HT min gain", opt["ht_gain"], lambda v: f"{v:g}pp", set_("ht_gain"),
                             values=(1., 2., 3., 5., 8., 10., 15., 20.))])
            self._section(box, "STYLE")
            box.pack_start(self._slider("Safe to Push", opt["push"],
                                        lambda v: "auto" if abs(v) < .01 else f"{'push' if v > 0 else 'safe'} {abs(v):.2f}",
                                        set_("push"), -.5, .5, .05,
                                        tooltip="Middle: follows your form. Push: higher-pp, lower-accuracy maps. "
                                                "Safe: comfortable ones."), False, False, 0)
            self._section(box, "REPEATS")
            box.pack_start(self._slider("Repeat cooldown", opt["cooldown_days"], lambda v: f"{v:.0f} days",
                                        set_("cooldown_days"), 1, 14, 1,
                                        tooltip="Offers and plays older than this no longer hold a map back."),
                           False, False, 0)
        done = Gtk.Button(label="Done", halign=Gtk.Align.END, margin_top=10)
        done.connect("clicked", lambda _b: self.gear_btn.set_active(False))
        box.pack_start(done, False, False, 0)
        sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
        sw.add(box)
        return sw

    def _recs_window(self):
        """Its own window, so the main one can stay an OBS source: selected map, pick, list."""
        win = self.recs = Gtk.Window(title="ManiaScope — Playlist", default_width=self._recs_size[0],
                                    default_height=self._recs_size[1])
        bar = Gtk.HeaderBar(show_close_button=True)
        bar.set_custom_title(Gtk.Box())   # no title text: the controls need the width
        for button in (self.next_btn, self.skip_btn, self.track_btn):
            if button is not None:
                bar.pack_start(button)
        # In-tab settings page, not a popover: the PP tab has ~8 controls (GNOME HIG keeps popovers small).
        # The gear sits in the current tab's label (user 2026-10-30) and moves with the tab.
        self.gear_btn = Gtk.ToggleButton(image=Gtk.Image.new_from_icon_name("emblem-system-symbolic", Gtk.IconSize.MENU),
                                         relief=Gtk.ReliefStyle.NONE, can_focus=False)
        self.gear_btn.set_tooltip_text("Settings for this tab")
        self.gear_btn.connect("toggled", self._toggle_settings)
        self._tab_labels = []
        bar.show_all()
        win.set_titlebar(bar)
        win.set_title("ManiaScope — Playlist")   # after the header bar, else Wayland shows "skillsets.py" (KWin rules match it)
        win.connect("configure-event", self._on_recs_size)
        win.set_transient_for(self)
        win.connect("delete-event", lambda *_: (self.recs_btn.set_active(False), True)[1])
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=14)
        # Content never sizes the window: KWin grew it to each new natural/minimum width (user 2026-10-30, twice).
        # Only the user's resize (and the saved size) sets it; too-wide content scrolls instead.
        outer = Gtk.ScrolledWindow(propagate_natural_width=False, propagate_natural_height=False)
        outer.add(box)
        outer.show()
        win.add(outer)
        # Wrap, never ellipsize: a narrow window must still show every word (user 2026-10-30).
        line = lambda **kw: Gtk.Label(xalign=0, wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR, max_width_chars=50, **kw)
        self.card_lbl = FixedBlock(1, 2, 1)
        self.card_lbl.set_parts(["<small>select a map in osu!</small>"])
        box.pack_start(self.card_lbl, False, False, 0)
        self.result_lbl = FixedBlock(1, 1, 2)          # shown from the first finished play
        box.pack_start(self.result_lbl, False, False, 0)
        # The one status line and bar for the whole window, at the bottom (user 2026-10-30: "Analysing" three times).
        self._status_text, self._bg_progress = "", None
        self.rec_status = line(no_show_all=True)
        self.rec_status.get_style_context().add_class("dim-label")
        self.rec_status.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.85"))
        box.pack_end(self.rec_status, False, False, 0)
        # A first run waits minutes (download, analysis): it must show what and how far, never just "loading".
        self.rec_progress = Gtk.ProgressBar(no_show_all=True)
        box.pack_end(self.rec_progress, False, False, 0)
        self._rec_pinned = {}              # lines later statuses must not hide: why nothing loads, slow mode
        self._rec_notebook = Gtk.Notebook()
        box.pack_start(self._rec_notebook, True, True, 2)
        for mode in recommend.MODES:
            page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=6)
            key_btn = self._key_picker(mode)
            target = FixedBlock(1, 2)            # the picked map, once there is one
            page.pack_start(target, False, False, 0)
            sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
            listing = Gtk.ListBox(activate_on_single_click=True)
            listing.connect("row-activated", lambda _l, row, m=mode: self._activate_rec(m, row))
            placeholder = Gtk.Label(label=f"Loading {recommend.MODE_NAMES[mode]} suggestions…", wrap=True, max_width_chars=50,
                                    justify=Gtk.Justification.CENTER, margin=24)
            placeholder.get_style_context().add_class("dim-label")
            placeholder.show()
            listing.set_placeholder(placeholder)
            note = line()
            note.get_style_context().add_class("dim-label")
            note.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.85"))
            view = {"list": listing, "target": target, "note": note, "keys": key_btn,
                    "hover": False, "pending": None, "ready": False, "target_data": None,
                    "placeholder": placeholder, "analysing": None, "loaded": False,
                    "status_text": "", "name": recommend.MODE_NAMES[mode]}
            self._rec_views[mode] = view
            sw.connect("enter-notify-event", lambda *_a, m=mode: self._rec_hover(m, True))
            sw.connect("leave-notify-event", lambda *_a, m=mode: self._rec_hover(m, False))
            sw.add(listing)
            page.pack_start(sw, True, True, 0)
            page.pack_end(note, False, False, 0)
            stack = view["stack"] = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
            stack.add_named(page, "list")
            stack.add_named(self._settings_page(mode, key_btn), "settings")
            stack.show_all()
            tab = Gtk.Box(spacing=2)
            tab.pack_start(Gtk.Label(label=recommend.MODE_NAMES[mode]), False, False, 0)
            tab.show_all()
            self._tab_labels.append(tab)
            self._rec_notebook.append_page(stack, tab)
        self._rec_notebook.set_current_page(recommend.MODES.index(self._rec_mode))
        self._tab_labels[recommend.MODES.index(self._rec_mode)].pack_start(self.gear_btn, False, False, 0)
        self.gear_btn.show_all()       # its tab label was already shown: the gear was missing until a tab switch
        from stats_view import StatsView
        self._stats_open = False
        self.stats_view = StatsView(self._practice_from_stats)
        self.stats_view.show_all()
        self._rec_notebook.append_page(self.stats_view, Gtk.Label(label="Stats"))
        self._rec_notebook.connect("switch-page", self._switch_rec_page)
        box.show_all()
        self.recs_btn.connect("toggled", lambda b: self._present_recs() if b.get_active() else win.hide())

    def _toggle_settings(self, button):
        mode = recommend.MODES[self._rec_notebook.get_current_page()] if not self._stats_open else None
        if mode:
            self._rec_views[mode]["stack"].set_visible_child_name("settings" if button.get_active() else "list")

    def _on_rec(self, msg):
        """Messages from the recommendation worker (main thread)."""
        if self._closed:
            return False
        if msg["type"] == "stats":
            self.stats_view.update(msg["data"])
        elif msg["type"] == "status":
            if msg.get("pin"):
                self._rec_pinned[msg["pin"]] = msg["text"]
            elif msg.get("blocking"):
                self._rec_pinned["blocker"] = msg["text"]
            # A later import/replay note must not hide why there are no recommendations at all.
            text = "\n".join(dict.fromkeys(t for t in (*self._rec_pinned.values(), msg["text"]) if t))
            done, total = msg.get("progress") or (0, 0)
            self._bg_progress = (done, total) if total else None
            self._set_status(text)
            if "learning" in msg:          # Stats has no list to show meanwhile: it keeps its own banner
                eta = msg["text"].rpartition(" · ~")[2] if " · ~" in msg["text"] else ""
                learning = (f"Learning your level from your plays · {done:,} / {total:,} charts analysed"
                            + (f" · ~{eta}" if eta else "") if msg["learning"] and total else None)
                self.stats_view.wait.show_progress(learning and learning + " — your stats fill in as this runs",
                                                   done, total)
            for v in self._rec_views.values():
                v["status_text"] = text
                self._set_placeholder(v)
            if not self.next_btn.get_sensitive():   # a greyed-out Next says what it waits for
                self.next_btn.set_tooltip_text(text or None)
        elif msg["type"] == "pool":
            mode = msg.get("mode", "pp")
            if (msg.get("revision", 0) != self._rec_revision or mode not in self._rec_views
                    or msg.get("selection_id", 0) < self._selection_id):
                return False
            view = self._rec_views[mode]
            view["pending"] = msg
            if not view["hover"] or not view["ready"]:
                self._flush_pool(mode)
        elif msg["type"] == "result":
            self.result_lbl.set_parts(result_markup(msg), f"{msg['title']}\n{msg['text']}")
            if self._auto_next and self._rec_mode in msg.get("targets", ()) and self.next_btn.get_sensitive():
                self._next(False)
        elif msg["type"] == "history":
            if getattr(self, "_history_dialog", None):
                self._history_text.get_buffer().set_text(msg["text"])
        elif msg["type"] == "card":
            if 'path' in msg and (msg['path'], msg['rate']) != (self._obs.get('path'), self._obs.get('rate')):
                return False
            self._card_data = msg["c"]
            self._card_note = msg.get("note")
            self._show_card()
        elif msg["type"] == "target":
            mode = msg.get("mode", "pp")
            if (msg.get("revision", self._rec_revision) != self._rec_revision or mode not in self._rec_views
                    or msg.get("selection_id", 0) < self._selection_id):
                return False
            c = msg["c"]
            target = self._rec_views[mode]["target"]
            self._rec_views[mode]["target_data"] = c
            self._show_card()
            target.set_parts(target_markup(c, self._obs) if c else [GLib.markup_escape_text(msg.get("msg") or "")],
                             "\n".join(t for t in (msg.get("msg"), c and c.get("why"), msg.get("copy")) if t) or None)
            if msg.get("copy"):
                Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).set_text(msg["copy"], -1)
        return False

    def _show_card(self):
        c, pick = self._card_data, self._rec_views.get(self._rec_mode, {}).get("target_data")
        # A PP map picked from the NPS list reads like one in the PP list (user 2026-10-31).
        pp = c and pick and pick.get("purpose") == "nps_mix" and navigation.same_map(pick, md5=c["md5"], bid=c["bid"])
        self.card_lbl.set_parts(*card_markup(c, "pp" if pp else self._rec_mode, self._card_note))
        self._overlay_update()

    def _set_status(self, text):
        self._status_text = text
        self._bottom_status()

    def _bottom_status(self):
        """Status line + bar: the worker's status (its progress when it reports one), else this tab's list analysis."""
        view = None if self._stats_open else self._rec_views.get(self._rec_mode)
        analysing = view and not self._bg_progress and view["analysing"]
        text = "\n".join(t for t in (self._status_text, analysing and analysing[0]) if t)
        done, total = self._bg_progress or (analysing[1:] if analysing else (0, 0))
        self.rec_status.set_text(text.replace("\n", " · "))
        self.rec_status.set_tooltip_text(text or None)
        self.rec_status.set_visible(bool(text))
        self.rec_progress.set_visible(bool(total))
        self.rec_progress.set_fraction(min(1., done / total) if total else 0.)

    def _key_picker(self, mode):
        pop = Gtk.Popover()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3, margin=8)
        pop.add(box)
        all_btn = Gtk.Button(label="All supported modes (4K–10K)")
        box.pack_start(all_btn, False, False, 0)
        checks = self._key_checks[mode] = {}
        def changed(_button):
            if not self._keys_syncing:
                self._set_rec_keys(mode, [k for k, b in checks.items() if b.get_active()])
        for k in recdata.SUPPORTED_KEYS:
            b = Gtk.CheckButton(label=f"{k}K", active=k in self._rec_keys[mode])
            checks[k] = b
            box.pack_start(b, False, False, 0)
            b.connect("toggled", changed)
        def all_keys(_button):
            self._set_rec_keys(mode, recdata.SUPPORTED_KEYS)
        all_btn.connect("clicked", all_keys)
        box.show_all()
        button = Gtk.MenuButton(popover=pop)
        label = Gtk.Label(label=recdata.keys_label(self._rec_keys[mode]), ellipsize=Pango.EllipsizeMode.END, max_width_chars=22)
        button.add(label)
        button.set_tooltip_text("Choose any combination independently for this tab. All covers 4K–10K; empty selection stays empty.")
        return button

    def _skill_picker(self):
        from skill_picker import SkillPicker
        button = SkillPicker(self._rec_keys["skills"], self._practice_skills, self._set_practice_skills,
                             self._skill_weights, self._set_skill_weight, self._slider)
        self._skill_checks, self._any_skill_btn = button.checks, button.any_button
        return button

    def _set_practice_skills(self, selected):
        selected = skill_practice.normalize(selected)
        if selected == self._practice_skills:
            return
        self._practice_skills = selected
        self.practice_btn.set_selection(selected)
        view = self._rec_views["skills"]
        for row in view["list"].get_children():
            view["list"].remove(row)
        view["ready"], view["pending"], view["target_data"] = False, None, None
        view["target"].set_parts([])
        view["note"].set_text("Finding " + skill_practice.label(selected) + " maps at your current level…")
        self._sync_rec()

    def _set_skill_weight(self, key, value):
        self._skill_weights[key] = value
        self._sync_rec()

    def _auto_keys(self):
        """Until the first lazer import lands (or the player picks), recheck what they play."""
        played = not self._closed and self._keys_auto and recdata.played_keys(self.tracking_db)
        for m in sorted(self._keys_auto) if played else ():
            self._set_rec_keys(m, played)
        return bool(self._keys_auto) and not self._closed

    def _set_rec_keys(self, mode, keys):
        self._rec_keys[mode] = recdata.normalize_keys(keys)
        self._keys_auto.discard(mode)
        self._keys_syncing = True              # the picker shows keymodes set from outside it too
        for k, b in self._key_checks.get(mode, {}).items():
            b.set_active(k in self._rec_keys[mode])
        self._keys_syncing = False
        if mode == "skills":
            self.practice_btn.set_keys(self._rec_keys[mode])
            self._skill_checks, self._any_skill_btn = self.practice_btn.checks, self.practice_btn.any_button
        view = self._rec_views[mode]
        view["keys"].get_child().set_text(recdata.keys_label(self._rec_keys[mode]))
        for row in view["list"].get_children():
            view["list"].remove(row)
        view["ready"], view["pending"] = False, None
        if view.get("target_data") and view["target_data"]["keys"] not in self._rec_keys[mode]:
            view["target_data"] = None
            view["target"].set_parts([])
        view["note"].set_text("Refreshing…" if self._rec_keys[mode] else "Select at least one keymode")
        self._sync_rec()

    def _switch_rec_page(self, _notebook, _page, index):
        active = index == len(recommend.MODES)
        self.gear_btn.set_active(False)        # still the old page: its settings close
        if self.gear_btn.get_parent():
            self.gear_btn.get_parent().remove(self.gear_btn)
        if not active:
            self._tab_labels[index].pack_start(self.gear_btn, False, False, 0)
            self.gear_btn.show_all()
        old = self._stats_open
        self._stats_open = active
        if not active:
            self._set_rec_mode(recommend.MODES[index])
        self._bottom_status()
        if active != old and hasattr(self, "rec"):
            self.rec.put("stats", active)

    def _practice_from_stats(self, keys, group):
        self._set_rec_keys("skills", [keys])
        self._set_practice_skills([group])
        self._set_rec_mode("skills")
        self._rec_notebook.set_current_page(recommend.MODES.index("skills"))

    def _set_rec_mode(self, mode):
        if mode not in recommend.MODES or mode == self._rec_mode:
            return
        self._rec_mode = mode
        self._mode_items[mode].set_active(True)
        self._rec_notebook.set_current_page(recommend.MODES.index(mode))
        self._show_card()
        self._sync_rec()

    def _set_next_action(self, action):
        if action in ("auto", "copy", "link"):
            self._next_action = action
            self._sync_rec()

    def _toggle_overlay(self, item):
        on = item.get_active()
        if on == self._overlay:
            return
        if on:
            dlg = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.WARNING,
                                    buttons=Gtk.ButtonsType.OK_CANCEL, text="Turn on tosu's in-game overlay?")
            dlg.format_secondary_text(
                "tosu draws inside osu! by loading a DLL into the game, so fullscreen works. It downloads its overlay "
                "program the first time. tosu offers this openly, but osu! has no official statement on it, so use it "
                "at your own risk.\n\nAfter it starts: in osu!, press Ctrl+Shift+Space, right-click, add ManiaScope, "
                "then press Ctrl+Shift+Space again.")
            answer = dlg.run()
            dlg.destroy()
            if answer != Gtk.ResponseType.OK:
                item.set_active(False)
                return
        self._overlay = on
        self._queue_save()
        if self._tosu_proc and self._tosu_proc.poll() is None:
            self._tosu_proc.terminate()
            self._tosu_proc.wait()
            self._start_tosu()
            self._overlay_update()
        else:
            self._set_status("Restart tosu to apply the overlay setting")

    def _overlay_update(self):
        """What the main window shows, plus the expected accuracy and the pick, for the overlay page."""
        if not self._overlay:
            return
        shown = self._shown if self._shown and not self._pending else None
        res, chart = shown and shown[3], shown and shown[2]
        c, pick = self._card_data, self._rec_views.get(self._rec_mode, {}).get("target_data")
        state = {}
        if res:
            tl, rate = res.get("timeline") or [], shown[1]
            step = max(1, len(tl) // 120)
            state = dict(title=chart.title, artist=chart.artist, version=f"[{chart.version}]  {chart.creator}",
                         rating=res["scores"]["overall"], context=self._context, nps=nps(chart, rate),
                         ln=round(100 * res["ln_notes"] / max(1, res["notes"])),
                         skills=[dict(name=e["name"], rating=e["rating"]) for e in skill_calc.card(res)[:self._display["skills"]]],
                         timeline=[max(tl[i:i + step]) for i in range(0, len(tl), step)])
        if c and c.get("acc_mid") is not None:
            state["expected"] = f"~{100 * recdata.shown_acc(c['acc_mid']):.1f}%"
        if pick:
            rate = pick["rate"]
            state["pick"] = dict(title=pick.get("title", ""), expected=f"~{100 * recdata.shown_acc(pick['acc_mid']):.1f}%",
                                 set=f"{rate:.2f}× " + ("NM" if abs(rate - 1.) < .0005 else "HT" if rate < 1. else "DT"))
        tools_setup.write_overlay_state(state)

    def _set_keep_above(self, on):
        self._keep_above = on
        self.set_keep_above(on)
        self.recs.set_keep_above(on)
        self._queue_save()

    def _set_nps_focus(self, value):
        self._nps_focus = nps_playlist.normalize_focus(value)
        self._sync_rec()

    def _sync_rec(self):
        self._rec_revision += 1
        # Settings apply at once; this answers "how do I refresh?" (Less, 2026-10-31).
        view = self._rec_views.get(self._rec_mode)
        if view and "note" in view:
            view["note"].set_text("Applied · updating the list…")
        for view in self._rec_views.values():
            for row in view["list"].get_children():
                row.revision = self._rec_revision
        self.next_btn.set_label("Next · " + recommend.MODE_NAMES[self._rec_mode])
        if hasattr(self, "rec"):
            self.rec.put("configure", self._rec_mode, {k: list(v) for k, v in self._rec_keys.items()},
                         self._next_action, self._rec_revision, list(self._practice_skills), self._nps_focus,
                         dict(self._acc_targets), dict(self._web_bias), dict(self._pp_settings), dict(self._pp_mix),
                         dict(self._skill_weights))
        self._next_sensitive()
        self._queue_save()

    def _next_sensitive(self):
        view = self._rec_views.get(self._rec_mode, {})
        enabled = bool(self._rec_keys[self._rec_mode]) and bool(view.get("ready"))
        self.next_btn.set_label("Next · " + recommend.MODE_NAMES[self._rec_mode])
        self.next_btn.set_sensitive(enabled)
        self.skip_btn.set_sensitive(enabled)

    def _next(self, skip):
        self._flush_pool(self._rec_mode)
        rows = self._rec_views[self._rec_mode]["list"].get_children()
        if rows:
            self._activate_rec(self._rec_mode, rows[0], skip=skip)

    def _activate_rec(self, mode, row, *, skip=None):
        if (row.revision == self._rec_revision and row.cand["keys"] in self._rec_keys[mode]
                and (mode != "skills" or tuple(row.cand.get("practice_skills", ())) == self._practice_skills)):
            c = dict(row.cand, mode=mode, selection="list" if skip is None else "next")
            selected_at = time.time()
            result = navigation.perform(c, "copy" if self._live is not None else self._next_action)
            self._selection_id += 1
            # Navigation uses the completed snapshot now; the worker only logs
            # this selection and refreshes the remaining playlist afterward.
            if skip is None:
                self.rec.put("open", c, mode, self._rec_revision, result, selected_at, self._selection_id)
            else:
                self.rec.put("next", skip, mode, self._rec_revision, c, result, selected_at, self._selection_id)
            if result.get("unavailable"):
                self._set_status(result["msg"])
            else:
                self._on_rec(dict(type="target", mode=mode, revision=self._rec_revision,
                                  selection_id=self._selection_id, c=c, copy=result["copy"],
                                  msg=result["msg"] + f" · set {c['var']} manually"))
            for view in self._rec_views.values():
                for old in view["list"].get_children():
                    if recommend.event_key(old.cand) == recommend.event_key(c):
                        view["list"].remove(old)
                view["ready"] = bool(view["list"].get_children())
            self._next_sensitive()

    def _rec_hover(self, mode, hover):
        self._rec_views[mode]["hover"] = hover
        if not hover:
            self._flush_pool(mode)
        return False

    def _tracking_toggled(self, btn):
        enabled = btn.get_active()
        play = getattr(getattr(self, "watcher", None), "_play", None)
        recdata.set_tracking(self.tracking_db, enabled, start=play["t"] if play else None)
        btn.set_label("Tracking on" if enabled else "Tracking off")
        self.rec.put("tracking")
        self._set_status("Tracking resumed for the next play" if enabled else "Tracking paused — controller plays will be excluded")

    def _show_history(self, _button):
        if getattr(self, "_history_dialog", None):
            self._history_dialog.present()
            return
        dlg = self._history_dialog = Gtk.Dialog(title="Play & prediction history", transient_for=self,
                                               default_width=760, default_height=520)
        dlg.add_button("Close", Gtk.ResponseType.CLOSE)
        box = dlg.get_content_area()
        only = Gtk.CheckButton(label="Only unusually high or low results (±1.5σ)", margin=8)
        only.connect("toggled", lambda b: self.rec.put("history", b.get_active()))
        box.pack_start(only, False, False, 0)
        box.pack_start(Gtk.Label(label="Latest 100 plays; unusual results sorted by prediction error. Accuracy is lazer accuracy.",
                                 xalign=0, margin=8), False, False, 0)
        self._history_text = Gtk.TextView(editable=False, cursor_visible=False, wrap_mode=Gtk.WrapMode.WORD_CHAR,
                                         left_margin=8, right_margin=8)
        sc = Gtk.ScrolledWindow()
        sc.add(self._history_text)
        box.pack_start(sc, True, True, 0)
        dlg.connect("response", lambda d, _r: d.destroy())
        dlg.connect("destroy", lambda _d: setattr(self, "_history_dialog", None))
        dlg.show_all()
        self.rec.put("history")

    def _check_tosu(self):
        if self._obs["source"] != "tosu" or (self._tosu_proc is not None and self._tosu_proc.poll() is not None):
            self._start_tosu()
        return True

    @staticmethod
    def _set_placeholder(view):
        # Until a tab's first list arrives it is loading, not empty (PP sat at "Nothing to suggest yet" ~20 s).
        head = "Nothing to suggest yet." if view["loaded"] else f"Loading {view['name']} suggestions…"
        view["placeholder"].set_text(head + (f"\n\n{view['status_text']}" if view["status_text"] else ""))


    def _flush_pool(self, mode):
        view = self._rec_views[mode]
        msg, view["pending"] = view["pending"], None
        if not msg or msg["revision"] != self._rec_revision or msg.get("selection_id", 0) < self._selection_id:
            return
        if not view["loaded"]:
            view["loaded"] = True
            self._set_placeholder(view)
        for row in view["list"].get_children():
            view["list"].remove(row)
        if msg["shown"]:
            self._rec_pinned.pop("blocker", None)
        phase = {"warmup": "Warming up", "build": "Warmed up", "push": "Strong form", "recover": "Easing off"}[msg["phase"]]
        view["note"].set_text(f"{phase}{' · ' + msg['note'] if msg['note'] else ''}")
        view["note"].set_tooltip_text("\n".join(t for t in (view["note"].get_text(), msg["summary"]) if t))
        done, total = msg.get("progress") or (0, 0)
        view["analysing"] = ((f"Analysing maps · {done:,} / {total:,} · paused while you play", done, total)
                             if total else None)
        self._bottom_status()
        for c in msg["shown"]:
            m, sec = divmod(int(c["length"]), 60)
            length = f"{m}:{sec:02d}"
            if c.get("purpose") == "nps_mix" or (mode == "pp" and c["purpose"] != "warmup"):
                number = ("PP " if mode != "pp" else "") + f"+{c['gain']:.1f}pp · {100 * c['p_up']:.0f}%"
                sub = [c["var"], f"{c['stars']:.1f}★" if c.get("stars") else "",
                       f"best {c['best']:.0f}pp" if c.get("best") else "new", length]
            elif mode == "pp":
                number, sub = f"warm-up ~{100 * recdata.shown_acc(c['acc_mid']):.1f}%", [length]
            else:
                number = f"~{100 * recdata.shown_acc(c['acc_mid']):.1f}%"
                sub = ([f"{c['nps']:.1f} NPS"] if mode == "nps" else []) + [length, c.get("description") or c.get("skill") or ""]
            if not c.get("installed", True):
                sub.append("download" if mode != "pp" else "not installed")
            title = Gtk.Label(xalign=0, wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR, max_width_chars=40, hexpand=True)
            title.set_markup(f"<b>{c['rate']:.2f}×</b>  {_map_markup(c)}")
            key = Gtk.Label(xalign=1, valign=Gtk.Align.START)
            key.set_markup(f"<b>{GLib.markup_escape_text(number)}</b>")
            detail = Gtk.Label(xalign=0, wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR, max_width_chars=40)
            detail.set_markup(f"<small>{GLib.markup_escape_text(' · '.join(x for x in sub if x))}</small>")
            detail.get_style_context().add_class("dim-label")
            grid = Gtk.Grid(column_spacing=10, row_spacing=2, margin_top=7, margin_bottom=7, margin_start=4, margin_end=4)
            grid.attach(title, 0, 0, 1, 1)
            grid.attach(key, 1, 0, 1, 1)
            grid.attach(detail, 0, 1, 2, 1)
            row = Gtk.ListBoxRow(tooltip_text=c.get("why"))
            row.cand = c
            row.revision = self._rec_revision
            row.add(grid)
            view["list"].add(row)
        view["list"].show_all()
        view["ready"] = bool(msg["shown"])
        self._next_sensitive()

    # ---- helper lifecycle: only a tosu WE started is ever stopped ---------
    def _start_tosu(self):
        self._tosu_tried = time.monotonic()
        if not os.access(TOSU_BIN, os.X_OK):
            # Packaged builds fetch tosu once, unasked: rate tracking is part of "just works".
            if getattr(sys, "frozen", False) and not getattr(self, "_tosu_download", None):
                self._install_tosu()
            return
        if self._tosu_proc and self._tosu_proc.poll() is None:
            return                       # ours is still coming up
        try:
            urllib.request.urlopen(TOSU_URL, timeout=0.3).close()
            return                       # someone's tosu already answers
        except urllib.error.HTTPError:
            return
        except OSError:
            pass
        try:
            # tosu only takes these from its config file; without the first every start opens a browser tab
            env = {"OPEN_DASHBOARD_ON_STARTUP": "false"}
            if paths.WINDOWS:
                # Nothing here reads keys or live hit errors: tosu's 100 Hz gameplay loop only costs the game CPU.
                # Other overlays in the in-game layer may want them (tosu's own defaults).
                env.update(ENABLE_INGAME_OVERLAY=str(self._overlay).lower(),
                           ENABLE_KEY_OVERLAY=str(self._overlay).lower(),
                           PRECISE_DATA_POLL_RATE=10 if self._overlay else 100)
            tools_setup.set_tosu_env(TOSU_ENV, env)
            if self._overlay:
                tools_setup.install_overlay()
            os.makedirs(CACHE_DIR, exist_ok=True)
            out = open(os.path.join(CACHE_DIR, "tosu.log"), "w")
            self._tosu_proc = subprocess.Popen(
                [TOSU_BIN], cwd=os.path.dirname(TOSU_BIN), stdout=out,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            log_exc("start tosu")

    def _install_tosu(self):
        def run():
            try:
                tools_setup.install_tosu()
                GLib.idle_add(self._on_status, "tosu installed — rate tracking starts")
                GLib.idle_add(self._start_tosu)
            except Exception as exc:          # offline / GitHub unreachable: the log fallback still works
                log_exc("install tosu")
                GLib.idle_add(self._on_status, f"tosu download failed: {exc}"[:120])
        self._tosu_download = threading.Thread(target=run, daemon=True, name="tosu-install")
        self._tosu_download.start()

    # ---- persisted UI state ----------------------------------------------
    @staticmethod
    def _load_ui():
        try:
            with open(UI_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _on_configure(self, _w, _e):
        if self.get_mapped() and self.get_window() and not self.get_window().get_state() & (
                Gdk.WindowState.MAXIMIZED | Gdk.WindowState.FULLSCREEN | Gdk.WindowState.ICONIFIED):
            size = tuple(self.get_size())
            if min(size) > 1:
                if self.details_btn.get_active():
                    if size != self._details_size:
                        self._details_size = size
                        self._queue_save()
                elif not self._restoring_compact and size != self._win_size:
                    self._win_size = size
                    self._queue_save()
        return False

    def _on_moved(self, _w, _e):
        pos = tuple(self.get_position())
        if self.get_mapped() and pos != self._win_pos:
            self._win_pos = pos
            self._queue_save()
        return False

    def _present_recs(self):
        if self._recs_pos and not self.recs.get_mapped():
            self.recs.move(*self._recs_pos)
        self.recs.present()

    def _on_recs_size(self, win, _event):
        if win.get_mapped() and win.get_window() and not win.get_window().get_state() & (
                Gdk.WindowState.MAXIMIZED | Gdk.WindowState.FULLSCREEN | Gdk.WindowState.ICONIFIED):
            size, pos = tuple(win.get_size()), tuple(win.get_position())
            if min(size) > 1 and (size, pos) != (self._recs_size, self._recs_pos):
                self._recs_size, self._recs_pos = size, pos
                self._queue_save()
        return False

    def _queue_save(self):
        if self._closed:
            return
        if self._save_timer:
            GLib.source_remove(self._save_timer)
        self._save_timer = GLib.timeout_add(700, self._save_ui)

    def _save_ui(self):
        self._save_timer = 0
        try:
            os.makedirs(os.path.dirname(UI_FILE), exist_ok=True)
            with open(UI_FILE + ".tmp", "w", encoding="utf-8") as fh:
                json.dump({"mode": "auto" if self._auto else "manual",
                           "recommendation_mode": self._rec_mode, "next_action": self._next_action,
                           "auto_next": self._auto_next, "keep_above": self._keep_above, "overlay": self._overlay,
                           "pp_keys": list(self._rec_keys["pp"]), "nps_keys": None if "nps" in self._keys_auto else list(self._rec_keys["nps"]),
                           "skills_keys": None if "skills" in self._keys_auto else list(self._rec_keys["skills"]), "practice_skills": list(self._practice_skills),
                           "skill_weights": self._skill_weights,
                           "nps_focus": self._nps_focus, "web_bias": self._web_bias,
                           "nps_acc_target": self._acc_targets["nps"], "skills_acc_target": self._acc_targets["skills"],
                           "pp_settings": self._pp_settings, "pp_mix": self._pp_mix,
                           "display": self._display,
                           "rate": self._manual_rate,   # observed rates are never persisted
                           "details": self.details_btn.get_active(),
                           "w": self._win_size[0], "h": self._win_size[1],
                           "details_w": self._details_size[0], "details_h": self._details_size[1],
                           "recs_w": self._recs_size[0], "recs_h": self._recs_size[1], "recs_pos": self._recs_pos, "pos": self._win_pos,
                           "recs_open": self.recs_btn.get_active()}, fh)
            os.replace(UI_FILE + ".tmp", UI_FILE)
        except OSError:
            pass
        return False

    def _on_destroy(self, *_):
        if self._closed:
            return
        self._closed = True
        lifecycle("destroy")
        self._gen += 1
        with self._req_cv:
            self._req = None
            self._req_cv.notify_all()
        self.watcher.stop()
        self.rec.stop()
        GLib.source_remove(self._tosu_timer)
        self.tracking_db.close()
        if self._save_timer:
            GLib.source_remove(self._save_timer)
        self._save_ui()
        if self._tosu_proc and self._tosu_proc.poll() is None:
            self._tosu_proc.terminate()
        if self._web_proc and self._web_proc.poll() is None:      # paused, not killed: Resume continues
            import webmaps
            open(os.path.join(webmaps.WEB_DIR, "stop"), "w").close()
        self.recs.destroy()

    # ---- rate / selection state ------------------------------------------
    def _effective(self):
        """→ (path, rate, rate_known) for what should be on display now."""
        path = self._local or self._obs.get("path")
        if not self._auto:
            return path, self._manual_rate, True
        if self._local:
            return path, 1.0, True
        rate = self._obs.get("rate")
        return path, (rate if rate else 1.0), rate is not None

    def _sv(self):
        """Scroll changes count unless lazer's selection has Constant Speed."""
        return not (self._auto and not self._local and self._obs.get("cs"))

    def _set_spin(self, value):
        self.rate_spin.handler_block(self._spin_handler)
        self.rate_spin.set_value(value)
        self.rate_spin.handler_unblock(self._spin_handler)

    def _on_obs(self, obs):
        if self._closed:
            return False
        if obs.get("client") in recdata.CLIENTS and obs["client"] != self._obs.get("client"):
            self.rec.put("client", obs["client"])     # the worker ignores the one it already serves
        if (obs.get("path"), obs.get("rate"), obs.get("mods")) != \
                (self._obs.get("path"), self._obs.get("rate"), self._obs.get("mods")):
            self.rec.put("selected", obs.get("path"), obs.get("rate"), obs.get("mods"))
            self._on_rec({"type":"card", "c":None, "note":"Updating selected-map prediction…" if obs.get('path') else None})
        self._obs = obs
        for view in self._rec_views.values():
            if view.get("target_data"):
                view["target"].set_parts(target_markup(view["target_data"], obs), view["target"].get_tooltip_text())
        died = self._tosu_proc is not None and self._tosu_proc.poll() is not None
        if obs["source"] != "tosu" and (died or time.monotonic() - self._tosu_tried > 10):
            self._start_tosu()           # helper died or was never up: bring it back
        self._refresh()
        return False

    def _on_status(self, text):
        if not self._closed:
            self.status.set_text(text)
        return False

    def _refresh(self):
        """Reconcile controls + request analysis for the current effective state."""
        if self._closed:
            return
        path, rate, _known = self._effective()
        sv = self._sv()
        if self._auto:
            self._set_spin(rate)
        self.follow_btn.set_visible(bool(self._local))
        if not path:
            self._gen += 1            # invalidate anything in flight
            self._requested = None
            with self._req_cv:
                self._req = None
            self._shown = None
            self._render()
        elif self._shown and (self._shown[0], self._shown[1], self._shown[5]) == (path, rate, sv):
            # Returning to the already displayed chart must invalidate a newer
            # in-flight selection too. Otherwise its delayed result can replace
            # the chart the user has actually settled on.
            if self._requested is not None:
                self._gen += 1
                self._requested = None
                with self._req_cv:
                    self._req = None
            self._render()            # same analysis, context (AUTO/MANUAL) may differ
        else:
            # Repeated status observations while this selection is computing
            # must not invalidate its own result or enqueue the same chart again.
            try:
                stamp = os.stat(path).st_mtime_ns
            except OSError:
                stamp = None
            request_key = (path, stamp, rate, sv)
            if request_key == self._requested:
                self._update_status()
                return
            self._requested = request_key
            self._gen += 1
            from analysis_cache import selected
            hit = selected.get_cached(path, round(rate, 4), sv)
            with self._req_cv:
                self._req = None if hit else (self._gen, path, rate, sv)
                self._req_cv.notify()
            if not hit:
                self._render_pending(path, rate)
            if hit:
                chart, result = hit
                self._on_result((self._gen, path, rate, chart, result, None, sv))
        self._update_status()

    def _worker(self):
        while not self._closed:
            with self._req_cv:
                while self._req is None and not self._closed:
                    self._req_cv.wait()
                if self._closed:
                    return
                gen, path, rate, sv = self._req
                self._req = None
            try:
                chart, res = analyze(path, rate, sv, cancelled=lambda: self._closed or gen != self._gen)
                out = (gen, path, rate, chart, res, None, sv)
            except skill_calc.AnalysisCancelled:
                continue  # Start the newest pending map without publishing a stale result/error.
            except (skill_calc.ChartError, OSError, ValueError) as exc:
                out = (gen, path, rate, None, None, str(exc), sv)
            except Exception:
                log_exc(f"analyze {path}")
                out = (gen, path, rate, None, None, "internal error (see crash.log)", sv)
            if not self._closed and gen == self._gen:
                GLib.idle_add(self._on_result, out)

    def _on_result(self, out):
        gen, path, rate, chart, res, err, sv = out
        if not self._closed and gen == self._gen:  # includes callbacks queued before destruction
            self._requested = None
            lifecycle(f"selection complete generation={gen} rate={rate:.3f} sv={sv} error={bool(err)}")
            self._shown = (path, rate, chart, res, err, sv)
            self._render()
            self._update_status()
        return False

    def on_auto_toggled(self, btn):
        self._auto = btn.get_active()
        if not self._auto:
            self._manual_rate = round(self.rate_spin.get_value(), 2)
        self._queue_save()
        self._refresh()

    def on_rate_edited(self, spin):
        """Only user edits reach here (programmatic sets are blocked)."""
        self._manual_rate = round(spin.get_value(), 2)
        if self._auto:
            self._auto = False
            self.auto_btn.handler_block(self._auto_handler)
            self.auto_btn.set_active(False)
            self.auto_btn.handler_unblock(self._auto_handler)
        self._queue_save()
        self._refresh()

    def on_open(self, _btn):
        dlg = Gtk.FileChooserDialog(title="Open .osu", parent=self,
                                    action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Open", Gtk.ResponseType.OK)
        flt = Gtk.FileFilter()
        flt.set_name("osu! beatmaps")
        flt.add_pattern("*.osu")
        dlg.add_filter(flt)
        if dlg.run() == Gtk.ResponseType.OK:
            self._local = dlg.get_filename()
            self._refresh()
        dlg.destroy()

    def on_follow(self, _btn):
        self._local = None
        self._refresh()

    def on_details_toggled(self, btn):
        active = btn.get_active()
        if active and self.get_mapped():
            # Capture the user's compact geometry before Details asks GTK for
            # extra vertical space. Expanded resizes are tracked separately.
            size = tuple(self.get_size())
            if min(size) > 1:
                self._win_size = size
        elif not active and self.get_mapped():
            # Hiding the rows can emit one last allocation at the expanded
            # height. Ignore it until the explicit compact resize has landed.
            self._restoring_compact = True
        self.details.set_reveal_child(active)
        self.details.set_visible(active)
        if not active and self.get_mapped():
            GLib.idle_add(self._restore_compact_size)
        self._queue_save()

    def _restore_compact_size(self):
        if self._closed or self.details_btn.get_active():
            self._restoring_compact = False
            return False
        self.resize(*self._win_size)
        # Wayland applies resize asynchronously. Keep expanded allocations from
        # overwriting the compact geometry until the resize has had a moment to
        # settle, then enforce it once more in case a layout pass raced us.
        GLib.timeout_add(120, self._finish_compact_restore)
        return False

    def _finish_compact_restore(self):
        if self._closed or self.details_btn.get_active():
            self._restoring_compact = False
            return False
        self.resize(*self._win_size)
        self._restoring_compact = False
        self._queue_save()
        return False

    # ---- presentation -----------------------------------------------------
    def _render_pending(self, path, rate):
        """Instant feedback while a (long) chart is analysed: its own title from the file header,
        the old analysis cleared so it can't be read as the new map's."""
        meta = {}
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if line == "[HitObjects]":
                        break
                    k, sep, v = line.partition(":")
                    if sep and k in ("Title", "TitleUnicode", "Artist", "ArtistUnicode", "Version", "Creator", "CircleSize"):
                        meta.setdefault(k, v.strip())
        except OSError:
            pass
        self._pending = True
        title = meta.get("TitleUnicode") or meta.get("Title") or os.path.basename(path)
        self.title_lbl.set_markup(f"<b>{GLib.markup_escape_text(title)}</b>"
                                  f"  <small>{GLib.markup_escape_text(meta.get('ArtistUnicode') or meta.get('Artist', ''))}</small>")
        self.version_lbl.set_text(f"[{meta.get('Version', '')}]  {meta.get('Creator', '')}")
        self.ln_pct_lbl.set_text("")
        self.number_lbl.set('<span size="300%" weight="bold">…</span>', ("#555555",))
        self.nps_lbl.set()
        keys = meta.get("CircleSize", "").split(".")[0]
        self._context = (f"{keys}K  ·  " if keys else "") + f"{rate:.2f}×"
        self.context_lbl.set_text(self._context)
        self.dominant_lbl.set_text("Calculating…")
        self.others_lbl.set_text(" ")
        self.tags_lbl.set_text(" ")
        for pb, val, _name in self._bars.values():
            pb.set_fraction(0)
            val.set_text("")
        self.timeline.queue_draw()

    def _render(self):
        self._pending = False
        shown = self._shown
        res = shown[3] if shown else None
        err = shown[4] if shown else None
        if not res:
            self.title_lbl.set_text("" if not shown else os.path.basename(shown[0]))
            self.version_lbl.set_text("")
            self.ln_pct_lbl.set_text("")
            self.number_lbl.set('<span size="300%" weight="bold">–</span>', ("#555555",))
            self.nps_lbl.set()
            self.context_lbl.set_text("")
            self.dominant_lbl.set_text(err or "Select a mania map in osu!")
            self.others_lbl.set_text(" ")
            self.tags_lbl.set_text(" ")
            self._show_rows(7)
            for pb, val, _name in self._bars.values():
                pb.set_fraction(0)
                val.set_text("")
            self.timeline.queue_draw()
            return
        _path, rate, chart, _res, _err, _sv = shown
        sc = res["scores"]
        self.title_lbl.set_markup(f"<b>{GLib.markup_escape_text(chart.title or '?')}</b>"
                                  f"  <small>{GLib.markup_escape_text(chart.artist)}</small>")
        self.version_lbl.set_text(f"[{chart.version}]  {chart.creator}")
        self.ln_pct_lbl.set_markup(f"<small>LN {100 * res['ln_notes'] / max(1, res['notes']):.0f}%</small>")
        self.number_lbl.set(*rating_parts(sc["overall"]))
        v = nps(chart, rate)
        self.nps_lbl.set(*nps_parts(v)) if v else self.nps_lbl.set()
        _p, _r, known = self._effective()
        mode = (" local file" if self._local and self._auto else
                "" if self._auto and known else
                " (osu! rate unknown)" if self._auto else " manual")
        # card entries (archetypes replace their components); type size and opacity shrink with
        # distance from the top one
        names = skill_calc.skill_names(chart.keys)
        entries = skill_calc.card(res)
        dan = dans.label(chart.keys, sc["overall"], res["ln_notes"] / max(1, res["notes"]),
                         dans.vibro_runs(chart.notes, rate)) if self._display["dan"] else None
        # dan before the rate: when the line is cut, the rate note goes first
        self._context = f"{chart.keys}K" + (f"  ·  {dan}" if dan else "") + f"  ·  {rate:.2f}×{mode}" + (
            "  ·  SV ignored (Constant Speed)" if res.get("sv") == "ignored" else "")
        self._render_clock()
        dom, tags = skill_calc.describe(res)
        spans = []
        for e in entries:
            r = e["rating"] / entries[0]["rating"]
            f = min(1., (r - 0.3) / 0.7)        # a later entry can outrate the first (Bracket over Delay): alpha caps at 100 %
            spans.append(f'<span size="{70 + 55 * f:.0f}%" alpha="{55 + 45 * f:.0f}%"'
                         f'{' weight="bold"' if r >= 0.85 else ""}>{e["name"]} {e["rating"]:.1f}</span>')
        tip = [f'{e["name"]}: {" + ".join(names[p] for p in e["parts"])}' for e in entries if len(e["parts"]) > 1]
        if any("technical" in e["parts"] for e in entries):
            tip += ["Tech passages (song time):"] + [
                f'  {int(w["t"] // 60)}:{w["t"] % 60:04.1f}  pattern {w["pattern"]:.0%} / rhythm {w["rhythm"]:.0%}'
                f'{" — " + w["why"] if w["why"] else ""}' for w in res.get("tech_where", ())]
        self.dominant_lbl.set_tooltip_text("\n".join(tip) or None)
        self.dominant_lbl.set_markup(spans[0] if spans else "too sparse to characterise")
        self.others_lbl.set_markup("  ".join(spans[1:self._display["skills"]]) or " ")
        self.tags_lbl.set_text(" · ".join(tags) or " ")
        self._show_rows(chart.keys)
        self._overlay_update()
        for key in names:
            pb, val, _name = self._bars[key]
            pb.set_fraction(min(1.0, sc[key] / BAR_FULL))
            val.set_text(f"{sc[key]:.2f}")
        self.timeline.queue_draw()

    def _show_rows(self, keys):
        """Detail rows of this keymode's skills only (4K and 5K+ vocabularies differ)."""
        names = skill_calc.skill_names(keys)
        order = {k: i for i, k in enumerate(names)}
        for key, row in self._bars.items():
            for w in row:
                w.set_no_show_all(key not in names)
                w.set_visible(key in names)
                self._grid.child_set_property(w, "top-attach", order.get(key, len(order) + 1))
            if key in names:
                row[2].set_text(names[key])

    def _span(self):
        """→ (first note ms, last end ms, rate) of the shown chart, or None."""
        if self._pending or not self._shown or not self._shown[3] or len(self._shown[2].notes) < 2:
            return None
        notes = self._shown[2].notes
        return notes[0][0], max(e for _t, e, _c in notes), self._shown[1]

    def _on_score(self, score, path, fresh=False, attempt=None):
        if self._closed:
            return False
        if not recdata.score_allowed(self.tracking_db, score, attempt if fresh else None, exclude=True):
            self.status.set_text("play excluded: tracking was paused")
            return False
        self.rec.put("score", score, path, fresh, attempt)
        score = {k: v for k, v in score.items() if k != "mods_list"}
        db = feedback.load()
        if feedback.add_score(db, score, path, self.tracking_db):
            feedback.save(db)
            self.status.set_text(f"score saved for calibration: {score['accuracy']:.2f}% "
                                 f"{score['rank']} ({len(db['scores'])} total)")
        return False

    def _on_progress(self, live_ms):
        if self._closed:
            return False
        self._live = live_ms
        self._render_clock()
        self.timeline.queue_draw()
        return False

    def _render_clock(self):
        """Context line: chart length, or time left (real seconds at this rate) in play."""
        span, text = self._span(), self._context
        if span:
            t0, t1, rate = span
            playing = self._live is not None and not self._local
            left = max(0, t1 - max(t0, self._live if playing else t0)) / 1000 / rate
            text += f"  ·  {int(left) // 60}:{int(left) % 60:02d}" + (" left" if playing else "")
        self.context_lbl.set_text(text)
        self.context_lbl.set_tooltip_text(text)

    def _update_status(self):
        obs = self._obs
        if self._local:
            text = "local file — osu! selection ignored"
        elif obs["source"] == "tosu" and obs["path"]:
            text = f"{obs.get('client') or 'osu!'} via tosu ✓" + (f" — {obs['note']}" if obs.get("note") else "")
            if self._auto and obs["rate"] is None:
                text += " · showing 1.00×"
        elif obs["source"] == "log":
            text = f"lazer via log — {obs['note']}"
            if self._auto and obs["path"]:
                text += " · showing 1.00×, set the rate manually"
        else:
            text = obs["note"] + " (or ☰ → Open .osu…)"
        if self._shown and self._shown[2] and self._shown[2].keys != 7:
            text += "  ·  7K-first calibration"
        self.status.set_text(text)
        self.status.set_tooltip_text(text)

    def on_draw_timeline(self, area, cr):
        try:
            res = self._shown[3] if self._shown and not self._pending else None
            tl = res["timeline"] if res else None
            if not tl:
                return False
            w, h = area.get_allocated_width(), area.get_allocated_height()
            col = area.get_style_context().get_color(Gtk.StateFlags.NORMAL)
            step = max(1, len(tl) // max(1, w))       # one point per pixel at most
            pts = [max(tl[i:i + step]) for i in range(0, len(tl), step)]
            top = max(pts) * 1.05 or 1.0              # own peak: no dead band under the text
            cr.move_to(0, h)
            for i, v in enumerate(pts):
                cr.line_to(i * w / max(1, len(pts) - 1), h - h * v / top)
            cr.line_to(w, h)
            cr.close_path()
            cr.set_source_rgba(col.red, col.green, col.blue, 0.35)
            cr.fill_preserve()
            span = self._span()
            if self._live is not None and span and not self._local:
                x = w * min(1.0, max(0.0, (self._live - span[0]) / max(1, span[1] - span[0])))
                cr.save()
                cr.clip()
                cr.rectangle(0, 0, x, h)
                cr.set_source_rgba(0.21, 0.52, 0.89, 0.9)
                cr.fill()
                cr.restore()
                cr.set_source_rgba(0.21, 0.52, 0.89, 1)
                cr.rectangle(x - 1, 0, 2, h)
                cr.fill()
            cr.new_path()
        except Exception:
            log_exc("timeline")
        return False

    # ---- feedback ---------------------------------------------------------
    def on_save_map(self, _btn):
        if not self._shown or not self._shown[3]:
            return
        path, rate, chart = self._shown[:3]
        db = feedback.load()
        eid = feedback.add_map(db, path, chart, rate)
        feedback.save(db)
        self.status.set_text(f"saved: {feedback.label(db['maps'][eid])}")

    def on_compare(self, _btn):
        db = feedback.load()
        dlg = Gtk.Dialog(title="Saved maps & comparisons", parent=self, modal=True)
        dlg.set_default_size(560, 420)
        box = dlg.get_content_area()
        box.set_spacing(6)
        box.set_border_width(10)

        def combo():
            c = Gtk.ComboBoxText(hexpand=True)
            c.get_cells()[0].set_property("ellipsize", Pango.EllipsizeMode.END)
            return c

        ca, cb = combo(), combo()
        verdict = Gtk.ComboBoxText()
        for v in feedback.VERDICTS:
            verdict.append_text(v)
        note = Gtk.Entry(placeholder_text="optional note, e.g. “LN coordination”, “brief jack burst”")
        listing = Gtk.ListStore(str, str, str)   # kind, id, text
        view = Gtk.TreeView(model=listing, headers_visible=False)
        cell = Gtk.CellRendererText(ellipsize=Pango.EllipsizeMode.END)
        view.append_column(Gtk.TreeViewColumn("", cell, text=2))

        def fill():
            ida, idb = ca.get_active_id(), cb.get_active_id()
            for c in (ca, cb):
                c.remove_all()
                for eid, m in db["maps"].items():
                    c.append(eid, feedback.label(m))
            ids = list(db["maps"])
            ca.set_active_id(ida if ida in db["maps"] else ids[-1] if ids else None)
            cb.set_active_id(idb if idb in db["maps"] else ids[-2] if len(ids) > 1 else None)
            listing.clear()
            for j in db["judgments"]:
                a, b = db["maps"][j["a"]], db["maps"][j["b"]]
                listing.append(("j", f"{j['a']}|{j['b']}",
                                f"⚖ {feedback.label(a)}  vs  {feedback.label(b)} → "
                                f"{j['verdict']}  {j.get('note', '')}"))
            for eid, m in db["maps"].items():
                listing.append(("m", eid, f"★ {feedback.label(m)}"))

        def on_record(_b):
            a, b, v = ca.get_active_id(), cb.get_active_id(), verdict.get_active_text()
            if a and b and a != b and v:
                feedback.judge(db, a, b, v, note.get_text().strip(), skill_calc.MODEL_VERSION)
                feedback.save(db)
                note.set_text("")
                fill()

        def on_remove(_b):
            model, it = view.get_selection().get_selected()
            if not it:
                return
            kind, key = model[it][0], model[it][1]
            if kind == "m":
                feedback.remove_map(db, key)
            else:
                pair = set(key.split("|"))
                db["judgments"] = [j for j in db["judgments"] if {j["a"], j["b"]} != pair]
            feedback.save(db)
            fill()

        grid = Gtk.Grid(column_spacing=6, row_spacing=4)
        grid.attach(Gtk.Label(label="A", xalign=0), 0, 0, 1, 1)
        grid.attach(ca, 1, 0, 2, 1)
        grid.attach(Gtk.Label(label="B", xalign=0), 0, 1, 1, 1)
        grid.attach(cb, 1, 1, 2, 1)
        grid.attach(verdict, 1, 2, 1, 1)
        rec = Gtk.Button(label="Record")
        rec.connect("clicked", on_record)
        grid.attach(rec, 2, 2, 1, 1)
        grid.attach(note, 1, 3, 2, 1)
        box.add(Gtk.Label(xalign=0, wrap=True, label=(
            "Save maps with ⋯ → Save this map in the main window (the rate is part of the entry), then "
            "compare two of them here. Recording a pair again corrects it.")))
        box.add(grid)
        sw = Gtk.ScrolledWindow(vexpand=True)
        sw.add(view)
        box.add(sw)
        rm = Gtk.Button(label="Remove selected", halign=Gtk.Align.END)
        rm.connect("clicked", on_remove)
        box.add(rm)
        fill()
        dlg.show_all()
        dlg.run()
        dlg.destroy()


def main():
    import sys
    import faulthandler
    faulthandler.enable(all_threads=True)  # the packaged exe points stderr at maniascope.log first
    sys.excepthook = lambda *a: log_exc("uncaught", a)
    threading.excepthook = lambda args: log_exc(f"thread:{args.thread.name}",
                                               (args.exc_type, args.exc_value, args.exc_traceback))
    # Gtk.Application is single-instance: a second launch just raises this window
    # The playlist builder is pure-Python CPU for minutes after each score; at the default 5 ms GIL
    # switch interval every GTK callback and syscall of the viewer waited a full slice behind it
    # (G835LX 2026-10-02: "several seconds to load anything"; 512 cache stats 921 ms → 38 ms at 1 ms).
    sys.setswitchinterval(.001)
    app = Gtk.Application(application_id="io.github.robby250.ManiaScope")

    def activate(app):
        if getattr(app, "viewer", None) is not None and not app.viewer._closed:
            app.viewer.show()
            app.viewer.present()
            lifecycle("existing viewer activated")
            return
        try:
            import selected_analysis
            selected_analysis.preserve_runtime()
        except OSError:
            log_exc("preserve prediction runtime")
        # App-set values outrank the desktop's (XSETTINGS, settings.ini); Arial stands in where
        # Liberation Sans is missing (same metrics; Windows has no Liberation).
        Gtk.Settings.get_default().set_properties(
            gtk_theme_name="ManiaScope", gtk_icon_theme_name="ManiaScope",
            gtk_font_name="Liberation Sans, Arial 10")
        win = ManiaScopeWindow()
        app.viewer = win
        app.add_window(win)
        # SIGTERM closes like the window button, so a tosu we started is stopped too
        def terminate():
            lifecycle("SIGTERM")
            win.destroy()
            return False
        if not paths.WINDOWS:  # no unix signals in Windows GLib; the call crashed every launch there
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, 15, terminate)
        win.show_all()
        lifecycle("opened")

    def start(app):
        try:
            activate(app)
        except Exception:
            # Otherwise the process lingers with no window (Windows friends, v0.3.8)
            log_exc("startup")
            dlg = Gtk.MessageDialog(message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.CLOSE,
                                    text="ManiaScope could not start",
                                    secondary_text=f"Please send the log files in\n{CACHE_DIR}")
            dlg.run()
            app.quit()

    app.connect("activate", start)
    app.run(None)
    lifecycle("application loop exited")


if __name__ == "__main__":
    main()

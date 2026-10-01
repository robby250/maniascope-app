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
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gdk, Gtk, GLib, Pango, PangoCairo  # noqa: E402
import cairo  # noqa: E402

import functools  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

import native_backend
native_backend.activate()

import feedback  # noqa: E402
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
            delay = (0.3 if self._live is not None else (0.01 if obs.get("path") else 0.1)) if tosu else 1.0
            self._halt.wait(delay)

    # -- tosu ---------------------------------------------------------------
    def _poll_tosu(self):
        try:
            with urllib.request.urlopen(TOSU_URL, timeout=0.6) as resp:
                d = json.load(resp)
        except urllib.error.HTTPError:
            return {"source": "tosu", "path": None, "rate": None,
                    "note": "tosu is running but has not found osu!lazer"}
        except (OSError, ValueError):
            return None   # tosu not reachable → log fallback
        try:
            if d.get("error") or "beatmap" not in d:
                return {"source": "tosu", "path": None, "rate": None,
                        "note": "tosu is running but has not found osu!lazer"}
            self._track(d["state"]["name"].lower(), d)
            if d["state"]["name"] == "resultScreen":
                self._note_score(d)
                if self._last and self._last.get("source") == "tosu":
                    return self._last   # a past score's mods are not the selection
            rel = d["files"]["beatmap"]
            path = None
            for root in (d["folders"].get("songs"), LAZER_FILES):
                if rel and root and os.path.isfile(os.path.join(root, rel)):
                    path = os.path.join(root, rel)
                    break
            rate, note = rate_from_mods(d["play"].get("mods"))
            cs = feedback.constant_speed(d["play"].get("mods"))
            if d["state"]["name"].lower() == "play":
                self._live = d["beatmap"]["time"]["live"]
            raw_mods = d["play"].get("mods")
            mod_list = raw_mods.get("array") if isinstance(raw_mods, dict) else raw_mods
            return {"source": "tosu", "path": path, "rate": rate, "cs": cs, "mods": mod_list,
                    "bid": d["beatmap"].get("id"),
                    "note": note if path else "selected map file not found"}
        except (KeyError, TypeError, AttributeError):
            return {"source": "tosu", "path": None, "rate": None,
                    "note": "unexpected tosu payload"}

    def _track(self, state, d):
        """Attempt lifecycle: start on entering gameplay (a jump back in time is a restart), abort on
        leaving it anywhere but the results screen; the result shown next belongs to this attempt."""
        rel, tm = d["files"]["beatmap"], d["beatmap"]["time"]
        if state == "play" and rel:
            live, first, last = tm.get("live") or 0, tm.get("firstObject") or 0, tm.get("lastObject") or 0
            p = self._play
            if p is None or p["sha"] != os.path.basename(rel) or live + 1500 < p["pos"]:
                # a jump back within the first seconds is gameplay taking over from the song-select preview
                # position, not a restart: the attempt starts again without an abort
                if p and (time.time() - p["t"] > 5 or p["sha"] != os.path.basename(rel)):
                    self.on_attempt("void" if self._disconnected else "abort", dict(p, end=time.time()))
                mods = d["play"].get("mods")
                root = d["folders"].get("songs") or LAZER_FILES
                p = self._play = {"sha": os.path.basename(rel), "path": os.path.join(root, rel),
                                  "rate": rate_from_mods(mods)[0], "t": time.time(), "pos": live, "progress": 0.0,
                                  "played_seconds": 0.,
                                  "mods": mods.get("array") if isinstance(mods, dict) else None}
                self.on_attempt("start", dict(p))
            delta = live - max(first, p["pos"])
            if 0 < delta < 2000 and p.get("rate"):
                p["played_seconds"] = p.get("played_seconds", 0.) + delta / 1000 / p["rate"]
            p["partial_stats"] = d.get("play", {}).get("hits")
            p["pos"], p["progress"] = live, max(0.0, min(1.0, (live - first) / max(1, last - first)))
        elif self._play:
            p, self._play = self._play, None
            if state == "resultscreen":
                self._done = p
            else:
                self.on_attempt("void" if self._disconnected else "abort", dict(p, end=time.time()))
        self._disconnected = False

    def _note_score(self, d):
        """Hand the user's own mania result to the UI; fresh only right after its own gameplay (a replay or
        an old result being viewed is history, not today's form)."""
        rs, rel = d.get("resultsScreen") or {}, d["files"]["beatmap"]
        name = rs.get("playerName")
        if not rel or not name or name != (d.get("profile") or {}).get("name") \
                or str((rs.get("mode") or {}).get("name", "")).lower() != "mania":
            return
        score = {"sha256": os.path.basename(rel), "rate": rate_from_mods(rs.get("mods"))[0],
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
            self.on_score(score, os.path.join(d["folders"].get("songs") or LAZER_FILES, rel), fresh, att)

    # -- log fallback -------------------------------------------------------
    _lazer_running = staticmethod(recommend.lazer_running)

    def _poll_log(self):
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
    meta = c.get("search_meta") or {}
    title, version = c.get("title", ""), meta.get("Version")
    if meta.get("Title"):
        title = " — ".join(x for x in (meta.get("Artist"), meta["Title"]) if x)
    elif " [" in title and title.rstrip().endswith("]"):
        title, version = title.split(" [", 1)
        version = version.rstrip()[:-1]
    esc = GLib.markup_escape_text
    return f"<b>{esc(title.removesuffix('.osu'))}</b>" + (f"\n<small>{esc(version)} · {c['keys']}K</small>" if version else f" · {c['keys']}K")


def target_markup(c, obs):
    if not c:
        return ""
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
    expected = f"Expected ~{100*c['acc_mid']:.1f}%" if c.get("acc_mid") is not None else "Prediction pending"
    if same_chart and sha and sha != observed_sha:
        expected = "Updated chart selected · see the current prediction above"
    return ("<small>RECOMMENDED</small>\n" + _map_markup(c)
            + f'\n<span foreground="{colour}" size="large" weight="bold">{mark} SET {rate:.2f}× · {wanted}</span>'
            + f"\n<small>{expected}</small>")


def card_markup(c, nps_mode=False, skills_mode=False, note=None):
    """The selected map's expectation, as the recommendation list words it."""
    if not c:
        return "<small>" + GLib.markup_escape_text(note or "Select a map to see its prediction") + "</small>"
    m, sec = divmod(int(c["length"]), 60)
    acc = f"~{100 * c['acc_mid']:.1f}%   ·   usual {100 * c['acc_lo']:.1f}–{100 * c['acc_hi']:.1f}%"
    if not c["ranked"]:
        pp = "no pp (unranked or unranked rate)"
    else:
        pp = (f"{c['pp_lo']:.0f}–{c['pp_hi']:.0f}pp · " + (f"best {c['best']:.0f} · " if c["best"] else "")
              + f"+{c['gain']:.1f}pp expected · {100 * c['p_up']:.0f}% to {'beat' if c['best'] else 'count'}")
    if nps_mode:
        pp = f"{c['nps']:.1f} raw NPS" if c.get("nps") else ""
    if skills_mode:
        pp = "Whole-map prediction · regular skill practice"
    skill = skill_calc.NAMES.get(c.get("skill"), c.get("skill") or "")
    return ("<small>SELECTED IN OSU!</small>\n" + _map_markup(c)
            + f"\n<b>Expected {acc}</b>"
            + f"\n<small>{c.get('rate', 1.):.2f}× · {m}:{sec:02d} · {GLib.markup_escape_text(skill)}</small>"
            + (f"\n<small>{GLib.markup_escape_text(pp)}</small>" if pp else ""))


class ManiaScopeWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="ManiaScope")
        ui = self._load_ui()
        self._rec_mode = ui.get("recommendation_mode", "pp")
        if self._rec_mode not in recommend.MODES:
            self._rec_mode = "pp"
        self._rec_keys = {"pp": recdata.normalize_keys(ui.get("pp_keys")),
                          "nps": recdata.normalize_keys(ui.get("nps_keys", [7])),
                          "skills": recdata.normalize_keys(ui.get("skills_keys", [7]))}
        self._practice_skills = skill_practice.normalize(ui.get("practice_skills"))
        self._nps_focus = nps_playlist.normalize_focus(ui.get("nps_focus", .5))
        self._acc_targets = {m: nps_playlist.normalize_target(ui.get(f"{m}_acc_target", nps_playlist.DEFAULT_TARGET))
                             for m in ("nps", "skills")}
        self._display = dict(DISPLAY_DEFAULTS, **{k: v for k, v in (ui.get("display") or {}).items()
                                                  if k in DISPLAY_DEFAULTS})
        self._next_action = ui.get("next_action", "auto")
        self._auto_next = ui.get("auto_next", True)
        if self._next_action not in ("auto", "copy", "link"):
            self._next_action = "auto"
        self._rec_revision, self._rec_views, self._card_data = 0, {}, None
        self._card_note = None
        self._selection_id = 0
        # `w`/`h` are the compact viewer geometry. Details may temporarily grow
        # the window, but collapsing must return to this size rather than letting
        # the timeline consume the space that the skill rows used.
        self._win_size = (int(ui.get("w", 400)), int(ui.get("h", 230)))
        self._details_size = (int(ui.get("details_w", self._win_size[0])),
                              int(ui.get("details_h", self._win_size[1])))
        self._restoring_compact = False
        self._recs_size = (int(ui.get("recs_w", 620)), int(ui.get("recs_h", 760)))
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
        self._tosu_tried = 0.0
        self._live = None           # playback position (ms) while lazer is in gameplay
        self._context = ""
        self.tracking_db = recdata.connect()

        bar = Gtk.HeaderBar(show_close_button=True)
        bar.set_custom_title(Gtk.Box())   # no title text: the controls need the width
        self.next_btn = Gtk.Button(label="Next · " + recommend.MODE_NAMES[self._rec_mode])
        self.next_btn.get_style_context().add_class("suggested-action")
        self.next_btn.set_tooltip_text("Pick from the active PP/NPS/Skills tab. Auto copies a local song-select search; set rate manually.")
        self.next_btn.connect("clicked", lambda _b: self._next(False))
        self.next_btn.set_sensitive(False)
        bar.pack_start(self.next_btn)
        self.recs_btn = Gtk.ToggleButton()
        self.recs_btn.add(Gtk.Image.new_from_icon_name("view-list-bullet-symbolic", Gtk.IconSize.BUTTON))
        self.recs_btn.set_tooltip_text("Recommendations")
        bar.pack_start(self.recs_btn)
        skip = Gtk.Button.new_from_icon_name("media-skip-forward-symbolic", Gtk.IconSize.BUTTON)
        skip.set_tooltip_text("Skip: not this map (avoided for about 12 h)")
        skip.connect("clicked", lambda _b: self._next(True))
        self.skip_btn = skip
        skip.set_sensitive(False)
        bar.pack_start(skip)
        if tracking_switch(self.tracking_db):
            self.track_btn = Gtk.ToggleButton(label="Tracking on", active=recdata.tracking_enabled(self.tracking_db))
            self.track_btn.set_label("Tracking on" if self.track_btn.get_active() else "Tracking off")
            self.track_btn.set_tooltip_text("Turn off before controller play. Paused plays stay excluded after imports and restarts. Resume for the next play.")
            self.track_btn.connect("toggled", self._tracking_toggled)
            bar.pack_start(self.track_btn)
        bar.pack_end(self._overflow())
        self.set_titlebar(bar)
        self.set_title("ManiaScope")   # taskbar name; the header bar shows none

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
        self.follow_btn = Gtk.Button(label="Follow lazer", no_show_all=True,
                                     relief=Gtk.ReliefStyle.NONE)
        self.follow_btn.connect("clicked", self.on_follow)
        foot.pack_end(self.follow_btn, False, False, 0)
        foot.set_margin_bottom(4)
        root.pack_start(foot, False, False, 0)

        self._apply_display()
        self._render()
        threading.Thread(target=self._worker, daemon=True).start()
        self._start_tosu()
        self.rec = recommend.Worker(lambda m: GLib.idle_add(self._on_rec, m), busy=lambda: self._live is not None)
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

        self.auto_btn = Gtk.CheckMenuItem(label="Follow lazer's rate (Auto)", active=self._auto)
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
        item("Reset learned NPS taste", lambda _m: self.rec.put("reset_taste"), "Keep scores, session and map history; forget only learned taste")
        item("Import lazer scores now", lambda _m: self.rec.put("import"))
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

    def _recommended_stars(self):
        """Nomod ★ range covering the player's recent plays at the usual 0.9–1.2× rates, or None."""
        try:
            rows = self.tracking_db.execute(
                "SELECT info FROM events WHERE kind='start' ORDER BY id DESC LIMIT 400").fetchall()
        except Exception:
            return None
        stars = sorted(f["stars"] for f in ((json.loads(r[0]).get("features") or {}) for r in rows)
                       if f.get("stars") and f.get("keys") in self._rec_keys["nps"])
        if len(stars) < 30:
            return None
        lo, hi = stars[len(stars) // 10], stars[len(stars) * 9 // 10]
        return round(lo / 1.2 * 2) / 2, round(hi / .9 * 2) / 2

    def _web_charts_row(self):
        """Settings → website charts: download chart text for maps you don't have, by nomod ★ range."""
        import webmaps
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        row = Gtk.Box(spacing=6)
        row.pack_start(Gtk.Label(label="Website charts ★"), False, False, 0)
        saved = None
        try:
            saved = webmaps._kv(webmaps._db(), "stars")
        except Exception:
            pass
        recommended = self._recommended_stars()
        lo_v, hi_v = saved or recommended or (5., 10.)
        spins = []
        for value in (lo_v, hi_v):
            spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(value=value, lower=0, upper=15, step_increment=.5),
                                  digits=1, numeric=True)
            spins.append(spin)
            row.pack_start(spin, False, False, 0)
        button = Gtk.Button(label="Download / update")
        row.pack_end(button, False, False, 0)
        box.pack_start(row, False, False, 0)
        note = Gtk.Label(xalign=0, wrap=True, max_width_chars=54)
        note.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.85"))
        note.set_text((f"Recommended from your plays: {recommended[0]:g}–{recommended[1]:g}★. " if recommended else "")
                      + "Only chart text (no audio) for maps you don't have; runs in the background at low priority.")
        box.pack_start(note, False, False, 0)
        log = os.path.join(CACHE_DIR, "webmaps.log")

        def poll():
            proc = self._web_proc
            try:
                with open(log, encoding="utf-8", errors="replace") as fh:
                    last = (fh.read().strip().splitlines() or [""])[-1]
            except OSError:
                last = ""
            if proc is not None and proc.poll() is None:
                note.set_text("Website charts: " + last[:90])
                return True
            note.set_text("Website charts: done — restart ManiaScope to use them" if proc and proc.returncode == 0
                          else f"Website charts stopped: {last[:90]}")
            button.set_sensitive(True)
            return False

        def start(_b):
            lo, hi = sorted(s.get_value() for s in spins)
            keys = ",".join(str(k) for k in self._rec_keys["nps"]) or "7"
            argv = ([sys.executable, "--webmaps"] if getattr(sys, "frozen", False)
                    else [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "webmaps.py")])
            os.makedirs(CACHE_DIR, exist_ok=True)
            self._web_proc = subprocess.Popen(
                argv + ["crawl", "--keys", keys, "--stars", f"{lo:g}-{hi:g}"], stdout=open(log, "w"),
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            button.set_sensitive(False)
            GLib.timeout_add_seconds(5, poll)
        self._web_proc = None
        button.connect("clicked", start)
        return box

    def _acc_target_row(self, mode):
        """Settings: the displayed-accuracy target this tab's picks aim at."""
        row = Gtk.Box(spacing=8)
        row.pack_start(Gtk.Label(label="Accuracy target"), False, False, 0)
        value = Gtk.Label(width_chars=14, xalign=0)
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 88., 98., .5)
        scale.set_draw_value(False)
        scale.set_value(100 * self._acc_targets[mode])
        scale.set_tooltip_text("Maps and rates are chosen so the predicted accuracy lands near this.")
        def changed(s, initial=False):
            target = nps_playlist.normalize_target(round(s.get_value() * 2) / 200)
            value.set_text(f"{100 * target:.1f}%" + (" (default)" if target == nps_playlist.DEFAULT_TARGET else ""))
            if not initial and target != self._acc_targets[mode]:
                self._acc_targets[mode] = target
                self._sync_rec()
        scale.connect("value-changed", changed)
        changed(scale, True)
        row.pack_start(scale, True, True, 0)
        row.pack_end(value, False, False, 0)
        return row

    def _recs_window(self):
        """Its own window, so the main one can stay an OBS source: selected-map card, pick, list."""
        win = self.recs = Gtk.Window(title="ManiaScope — Next", default_width=self._recs_size[0],
                                    default_height=self._recs_size[1])
        win.connect("size-allocate", self._on_recs_size)
        win.set_transient_for(self)
        win.connect("delete-event", lambda *_: (self.recs_btn.set_active(False), True)[1])
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=14)
        win.add(box)
        self.card_lbl = Gtk.Label(xalign=0, wrap=True, selectable=True)
        self.card_lbl.set_markup("<small>select a map in lazer</small>")
        box.pack_start(self.card_lbl, False, False, 0)
        self.rec_status = Gtk.Label(xalign=0, wrap=True)
        self.rec_status.get_style_context().add_class("dim-label")
        self.rec_status.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.85"))
        box.pack_end(self.rec_status, False, False, 0)
        self.result_lbl = Gtk.Label(xalign=0, wrap=True, no_show_all=True)
        box.pack_start(self.result_lbl, False, False, 0)
        self._rec_notebook = Gtk.Notebook()
        box.pack_start(self._rec_notebook, True, True, 2)
        for mode in recommend.MODES:
            page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=6)
            # Settings stay collapsed: the playlist is what the tab is for.
            settings = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin_top=4)
            expander = Gtk.Expander(label="Settings")
            expander.add(settings)
            page.pack_start(expander, False, False, 0)
            top = Gtk.Box(spacing=8)
            top.pack_start(Gtk.Label(label="Keymodes", xalign=0), False, False, 0)
            key_btn = self._key_picker(mode)
            top.pack_start(key_btn, True, True, 0)
            settings.pack_start(top, False, False, 0)
            if mode in self._acc_targets:
                settings.pack_start(self._acc_target_row(mode), False, False, 0)
            if mode == "nps":
                preference = Gtk.Box(spacing=8)
                preference.pack_start(Gtk.Label(label="Variety"), False, False, 0)
                self.nps_focus_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0., 1., .05)
                self.nps_focus_scale.set_draw_value(False)
                self.nps_focus_scale.set_value(self._nps_focus)
                self.nps_focus_scale.set_tooltip_text("Choose more variety or prioritize the highest NPS. Both aim at the accuracy target.")
                self.nps_focus_scale.get_accessible().set_name("NPS playlist: variety to high NPS")
                self.nps_focus_scale.connect("value-changed", self._set_nps_focus)
                preference.pack_start(self.nps_focus_scale, True, True, 0)
                preference.pack_end(Gtk.Label(label="High NPS"), False, False, 0)
                settings.pack_start(preference, False, False, 0)
                settings.pack_start(self._web_charts_row(), False, False, 0)
            if mode == "skills":
                skill_row = Gtk.Box(spacing=8)
                skill_row.pack_start(Gtk.Label(label="Skills", xalign=0), False, False, 0)
                self.practice_btn = self._skill_picker()
                skill_row.pack_start(self.practice_btn, True, True, 0)
                page.pack_start(skill_row, False, False, 0)
            target = Gtk.Label(xalign=0, wrap=True, selectable=True, max_width_chars=54)
            target.set_tooltip_text("Rate guidance only. You remain free to choose any rate in osu!.")
            page.pack_start(target, False, False, 0)
            note = Gtk.Label(xalign=0, wrap=True, max_width_chars=54)
            note.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.85"))
            page.pack_start(note, False, False, 0)
            sw = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
            listing = Gtk.ListBox(activate_on_single_click=True)
            listing.connect("row-activated", lambda _l, row, m=mode: self._activate_rec(m, row))
            status = Gtk.Label(xalign=0, wrap=True, max_width_chars=54)
            status.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.85"))
            view = {"list": listing, "target": target, "note": note, "keys": key_btn,
                    "status": status, "hover": False, "pending": None, "ready": False, "target_data": None}
            self._rec_views[mode] = view
            sw.connect("enter-notify-event", lambda *_a, m=mode: self._rec_hover(m, True))
            sw.connect("leave-notify-event", lambda *_a, m=mode: self._rec_hover(m, False))
            sw.add(listing)
            page.pack_start(sw, True, True, 0)
            page.pack_end(status, False, False, 0)
            page.show_all()
            self._rec_notebook.append_page(page, Gtk.Label(label=recommend.MODE_NAMES[mode]))
        self._rec_notebook.set_current_page(recommend.MODES.index(self._rec_mode))
        from stats_view import StatsView
        self._stats_open = False
        self.stats_view = StatsView(self._practice_from_stats)
        self.stats_view.show_all()
        self._rec_notebook.append_page(self.stats_view, Gtk.Label(label="Stats"))
        self._rec_notebook.connect("switch-page", self._switch_rec_page)
        box.show_all()
        self.recs_btn.connect("toggled", lambda b: win.present() if b.get_active() else win.hide())

    def _on_rec(self, msg):
        """Messages from the recommendation worker (main thread)."""
        if self._closed:
            return False
        if msg["type"] == "stats":
            self.stats_view.update(msg["data"])
        elif msg["type"] == "status":
            view = self._rec_views.get(msg.get("mode"))
            (view["status"] if view else self.rec_status).set_text(msg["text"])
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
            self.result_lbl.set_markup(f"<b>Last play · {GLib.markup_escape_text(msg['title'])}</b>\n"
                                       f"{GLib.markup_escape_text(msg['text'])}")
            self.result_lbl.show()
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
            self.card_lbl.set_markup(card_markup(msg["c"], self._rec_mode == "nps", self._rec_mode == "skills", self._card_note))
        elif msg["type"] == "target":
            mode = msg.get("mode", "pp")
            if (msg.get("revision", self._rec_revision) != self._rec_revision or mode not in self._rec_views
                    or msg.get("selection_id", 0) < self._selection_id):
                return False
            c = msg["c"]
            target = self._rec_views[mode]["target"]
            self._rec_views[mode]["target_data"] = c
            target.set_markup(target_markup(c, self._obs) if c else GLib.markup_escape_text(msg.get("msg") or ""))
            target.set_tooltip_text(msg.get("copy") or (c.get("why") if c else None))
            self.rec_status.set_text(msg.get("msg") or "")
            if msg.get("copy"):
                Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).set_text(msg["copy"], -1)
        return False

    def _key_picker(self, mode):
        pop = Gtk.Popover()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3, margin=8)
        pop.add(box)
        all_btn = Gtk.Button(label="All supported modes (4K–10K)")
        box.pack_start(all_btn, False, False, 0)
        checks = {}
        changing = [False]
        def changed(_button):
            if not changing[0]:
                self._set_rec_keys(mode, [k for k, b in checks.items() if b.get_active()])
        for k in recdata.SUPPORTED_KEYS:
            b = Gtk.CheckButton(label=f"{k}K", active=k in self._rec_keys[mode])
            checks[k] = b
            box.pack_start(b, False, False, 0)
            b.connect("toggled", changed)
        def all_keys(_button):
            changing[0] = True
            for b in checks.values():
                b.set_active(True)
            changing[0] = False
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
        button = SkillPicker(self._rec_keys["skills"], self._practice_skills, self._set_practice_skills)
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
        view["target"].set_text("")
        view["note"].set_text("Finding " + skill_practice.label(selected) + " maps at your current level…")
        self._sync_rec()

    def _set_rec_keys(self, mode, keys):
        self._rec_keys[mode] = recdata.normalize_keys(keys)
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
            view["target"].set_text("")
        view["note"].set_text("Refreshing…" if self._rec_keys[mode] else "Select at least one keymode")
        self._sync_rec()

    def _switch_rec_page(self, _notebook, _page, index):
        active = index == len(recommend.MODES)
        old = self._stats_open
        self._stats_open = active
        if not active:
            self._set_rec_mode(recommend.MODES[index])
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
        self.card_lbl.set_markup(card_markup(self._card_data, mode == "nps", mode == "skills", self._card_note))
        self._sync_rec()

    def _set_next_action(self, action):
        if action in ("auto", "copy", "link"):
            self._next_action = action
            self._sync_rec()

    def _set_nps_focus(self, scale):
        self._nps_focus = nps_playlist.normalize_focus(scale.get_value())
        self._sync_rec()

    def _sync_rec(self):
        self._rec_revision += 1
        for view in self._rec_views.values():
            for row in view["list"].get_children():
                row.revision = self._rec_revision
        self.next_btn.set_label("Next · " + recommend.MODE_NAMES[self._rec_mode])
        if hasattr(self, "rec"):
            self.rec.put("configure", self._rec_mode, {k: list(v) for k, v in self._rec_keys.items()},
                         self._next_action, self._rec_revision, list(self._practice_skills), self._nps_focus,
                         dict(self._acc_targets))
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
                self.rec_status.set_text(result["msg"])
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
        self.rec_status.set_text("Tracking resumed for the next play" if enabled else "Tracking paused — controller plays will be excluded")

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

    def _flush_pool(self, mode):
        view = self._rec_views[mode]
        msg, view["pending"] = view["pending"], None
        if not msg or msg["revision"] != self._rec_revision or msg.get("selection_id", 0) < self._selection_id:
            return
        for row in view["list"].get_children():
            view["list"].remove(row)
        phase = {"warmup": "Session: warming up", "build": "Session: warmed up",
                 "push": "Session: strong form", "recover": "Session: easing off"}[msg["phase"]]
        view["note"].set_text(f"{phase}{' · ' + msg['note'] if msg['note'] else ''}")
        view["note"].set_tooltip_text(msg["summary"])
        for c in msg["shown"]:
            m, sec = divmod(int(c["length"]), 60)
            head = _map_markup(c)
            if mode == "nps":
                tail = (f"{c['nps']:.1f} NPS · expect ~{100*c['acc_mid']:.1f}% · {m}:{sec:02d} · {c.get('description') or c.get('skill') or ''}"
                        + ("" if c.get("installed", True) else " · not installed · download"))
            elif mode == "skills":
                acc = f"Expected ~{100*c['acc_mid']:.1f}%"
                if c.get("acc_lo") is not None and c.get("acc_hi") is not None:
                    acc += f" ({100*c['acc_lo']:.1f}–{100*c['acc_hi']:.1f})"
                tail = f"{acc} · {m}:{sec:02d} · {c.get('description') or c.get('skill') or ''}"
            elif c["purpose"] == "warmup":
                tail = f"warmup · ~{100 * c['acc_mid']:.1f}% · {m}:{sec:02d}"
            else:
                tail = (f"+{c['gain']:.1f}pp · ~{100 * c['acc_mid']:.1f}% · {c['pp_lo']:.0f}–{c['pp_hi']:.0f}pp · "
                        + (f"best {c['best']:.0f} · " if c["best"] else "") + f"{m}:{sec:02d}"
                        + ("" if c["installed"] else " · not installed"))
            lb = Gtk.Label(xalign=0, wrap=True, max_width_chars=60, margin_top=7, margin_bottom=7)
            lb.set_markup(f"{head}\n<b>{c['rate']:.2f}×</b>  <small>{GLib.markup_escape_text(tail)}</small>")
            lb.set_tooltip_text(c.get("why"))
            row = Gtk.ListBoxRow()
            row.cand = c
            row.revision = self._rec_revision
            row.add(lb)
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
            # tosu only takes this from its config file; without it every start opens a browser tab
            os.makedirs(os.path.dirname(TOSU_ENV), exist_ok=True)
            try:
                with open(TOSU_ENV, "r", encoding="utf-8") as fh:
                    env = fh.read()
            except OSError:
                env = ""
            quiet = re.sub(r"(?m)^OPEN_DASHBOARD_ON_STARTUP=.*$", "OPEN_DASHBOARD_ON_STARTUP=false", env)
            if "OPEN_DASHBOARD_ON_STARTUP=" not in quiet:
                quiet += "OPEN_DASHBOARD_ON_STARTUP=false\n"
            if quiet != env:
                with open(TOSU_ENV, "w", encoding="utf-8") as fh:
                    fh.write(quiet)
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

    def _on_recs_size(self, win, _allocation):
        if win.get_mapped() and win.get_window() and not win.get_window().get_state() & (
                Gdk.WindowState.MAXIMIZED | Gdk.WindowState.FULLSCREEN | Gdk.WindowState.ICONIFIED):
            size = tuple(win.get_size())
            if min(size) > 1 and size != self._recs_size:
                self._recs_size = size
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
                           "auto_next": self._auto_next,
                           "pp_keys": list(self._rec_keys["pp"]), "nps_keys": list(self._rec_keys["nps"]),
                           "skills_keys": list(self._rec_keys["skills"]), "practice_skills": list(self._practice_skills),
                           "nps_focus": self._nps_focus,
                           "nps_acc_target": self._acc_targets["nps"], "skills_acc_target": self._acc_targets["skills"],
                           "display": self._display,
                           "rate": self._manual_rate,   # observed rates are never persisted
                           "details": self.details_btn.get_active(),
                           "w": self._win_size[0], "h": self._win_size[1],
                           "details_w": self._details_size[0], "details_h": self._details_size[1],
                           "recs_w": self._recs_size[0], "recs_h": self._recs_size[1]}, fh)
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
        if (obs.get("path"), obs.get("rate"), obs.get("mods")) != \
                (self._obs.get("path"), self._obs.get("rate"), self._obs.get("mods")):
            self.rec.put("selected", obs.get("path"), obs.get("rate"), obs.get("mods"))
            self._on_rec({"type":"card", "c":None, "note":"Updating selected-map prediction…" if obs.get('path') else None})
        self._obs = obs
        for view in self._rec_views.values():
            if view.get("target_data"):
                view["target"].set_markup(target_markup(view["target_data"], obs))
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
            self.dominant_lbl.set_text(err or "Select a mania map in osu!lazer")
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
                " (lazer rate unknown)" if self._auto else " manual")
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
            f = (r - 0.3) / 0.7
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
            text = "local file — lazer selection ignored"
        elif obs["source"] == "tosu" and obs["path"]:
            text = "lazer via tosu ✓" + (f" — {obs['note']}" if obs.get("note") else "")
            if self._auto and obs["rate"] is None:
                text += " · showing 1.00×"
        elif obs["source"] == "log":
            text = f"lazer via log — {obs['note']}"
            if self._auto and obs["path"]:
                text += " · showing 1.00×, set the rate manually"
        else:
            text = obs["note"] + " — or Open a .osu"
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
    faulthandler.enable(all_threads=True)
    sys.excepthook = lambda *a: log_exc("uncaught", a)
    threading.excepthook = lambda args: log_exc(f"thread:{args.thread.name}",
                                               (args.exc_type, args.exc_value, args.exc_traceback))
    # Gtk.Application is single-instance: a second launch just raises this window
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
        win = ManiaScopeWindow()
        app.viewer = win
        app.add_window(win)
        # SIGTERM closes like the window button, so a tosu we started is stopped too
        def terminate():
            lifecycle("SIGTERM")
            win.destroy()
            return False
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, 15, terminate)
        win.show_all()
        lifecycle("opened")

    app.connect("activate", activate)
    app.run(None)
    lifecycle("application loop exited")


if __name__ == "__main__":
    main()

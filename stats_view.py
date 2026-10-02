"""Native GTK player profile. Rendering only; data comes from player_stats."""
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

import dans
import skill_practice


import math

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def month_name(m):
    return f"{MONTHS[int(m) % 12]} {2000 + int(m) // 12}"


def dan_ticks(keys, lo, hi):
    """[(stars, short name)] of the keymode's main dan series inside lo..hi."""
    tiers = dans._table().get(dans.SERIES.get((keys, "rice")), [])
    prefix = dans.SHORT.get(dans.SERIES.get((keys, "rice")), "")
    return [(math.exp(v), t.replace(prefix.strip(), "").strip() or t) for t, v in tiers if lo <= math.exp(v) <= hi]


def _ink(widget, alpha):
    c = widget.get_style_context().get_color(Gtk.StateFlags.NORMAL)
    return c.red, c.green, c.blue, alpha


def _accent(widget):
    found, c = widget.get_style_context().lookup_color("theme_selected_bg_color")
    return c if found else widget.get_style_context().get_color(Gtk.StateFlags.NORMAL)


class Scale:
    """Shared stars → x mapping so every skill row and the dan axis line up."""
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, max(hi, lo + .5)

    def x(self, value, width):
        return 6 + (width - 12) * min(1., max(0., (value - self.lo) / (self.hi - self.lo)))


class DanAxis(Gtk.DrawingArea):
    """Dan names above the skill rows, at the difficulty their songs sit."""
    def __init__(self, keys, scale):
        super().__init__(hexpand=True)
        self.keys, self.scale = keys, scale
        self.set_size_request(100, 18)
        self.connect("draw", self.draw)

    def draw(self, widget, cr):
        width = self.get_allocated_width()
        cr.set_font_size(9)
        last = -99
        for stars, name in dan_ticks(self.keys, self.scale.lo, self.scale.hi):
            x = self.scale.x(stars, width)
            extent = cr.text_extents(name)
            left = min(max(0., x - extent.width / 2), width - extent.width)     # edge names stay whole
            if left < last + 4:
                continue                        # crowded tiers: keep every other name readable
            cr.set_source_rgba(*_ink(self, .55))
            cr.move_to(left, 11)
            cr.show_text(name)
            cr.move_to(x, 13); cr.line_to(x, 18); cr.set_line_width(1); cr.stroke()
            last = left + extent.width
        return False


class SkillBar(Gtk.DrawingArea):
    """● your level, a faint band for its uncertainty, ○ this session (only once warm)."""
    def __init__(self, keys, row, scale, session):
        super().__init__(hexpand=True, valign=Gtk.Align.CENTER)
        self.keys, self.row, self.scale, self.session = keys, row, scale, session
        self.set_size_request(100, 22)
        self.connect("draw", self.draw)

    def draw(self, widget, cr):
        width, height = self.get_allocated_width(), self.get_allocated_height()
        y, x = height / 2, lambda v: self.scale.x(v, width)
        cr.set_source_rgba(*_ink(self, .10))
        cr.set_line_width(2); cr.move_to(6, y); cr.line_to(width - 6, y); cr.stroke()
        for stars, _name in dan_ticks(self.keys, self.scale.lo, self.scale.hi):
            cr.move_to(x(stars), y - 4); cr.line_to(x(stars), y + 4); cr.set_line_width(1); cr.stroke()
        r = self.row
        if r.get("value") is None:
            return False
        accent = _accent(self)
        cr.set_source_rgba(accent.red, accent.green, accent.blue, .25)
        cr.rectangle(x(r["low"]), y - 5, max(1., x(r["high"]) - x(r["low"])), 10); cr.fill()
        cr.set_source_rgba(accent.red, accent.green, accent.blue, 1.)
        cr.arc(x(r["value"]), y, 5.5, 0, 2 * math.pi); cr.fill()
        if self.session and r.get("current") is not None:
            cr.set_source_rgba(*_ink(self, .95))
            cr.set_line_width(2); cr.arc(x(r["current"]), y, 6.5, 0, 2 * math.pi); cr.stroke()
        return False


class HistoryChart(Gtk.DrawingArea):
    """Overall level per month, dan lines behind it, peak and today marked."""
    def __init__(self, keys, history, peak):
        super().__init__(hexpand=True)
        self.keys, self.history, self.peak = keys, history, peak
        self.set_size_request(200, 110)
        self.connect("draw", self.draw)

    def draw(self, widget, cr):
        h = self.history
        if len(h) < 2:
            return False
        width, height = self.get_allocated_width(), self.get_allocated_height()
        values = [v for _m, v in h]
        lo, hi = min(values) * .97, max(values) * 1.03
        m0, m1 = h[0][0], h[-1][0]
        x = lambda m: 8 + (width - 82) * (m - m0) / max(1, m1 - m0)
        y = lambda v: height - 16 - (height - 24) * (v - lo) / max(1e-6, hi - lo)
        cr.set_font_size(9)
        last = None
        for stars, name in sorted(dan_ticks(self.keys, lo, hi), reverse=True):
            if last is not None and y(stars) - last < 11:
                continue                        # crowded low tiers: every other line stays readable
            last = y(stars)
            cr.set_source_rgba(*_ink(self, .12)); cr.set_line_width(1)
            cr.move_to(8, y(stars)); cr.line_to(width - 68, y(stars)); cr.stroke()
            cr.set_source_rgba(*_ink(self, .55)); cr.move_to(width - 62, y(stars) + 3); cr.show_text(name)
        accent = _accent(self)
        cr.set_source_rgba(accent.red, accent.green, accent.blue, 1.); cr.set_line_width(2)
        for i, (m, v) in enumerate(h):
            (cr.line_to if i else cr.move_to)(x(m), y(v))
        cr.stroke()
        for (m, v), filled in ((self.peak, False), (h[-1], True)):
            cr.arc(x(m), y(v), 4, 0, 2 * math.pi)
            cr.fill() if filled else (cr.set_line_width(2), cr.stroke())
        cr.set_source_rgba(*_ink(self, .55))
        for m in (m0, m1):
            cr.move_to(x(m) - (0 if m == m0 else 40), height - 3); cr.show_text(month_name(m))
        return False


class StatsView(Gtk.Box):
    def __init__(self, practice):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=10)
        self.practice, self.data = practice, None
        self.expanded = set()
        top = Gtk.Box(spacing=8)
        self.keys = Gtk.ComboBoxText()
        self.keys.append("all", "All keymodes")
        for k in range(4, 11):
            self.keys.append(str(k), f"{k}K")
        self.keys.set_active_id("7")
        self.keys.connect("changed", lambda *_: self.render())
        top.pack_start(self.keys, False, False, 0)
        label = Gtk.Label(label="PLAYER STATS", xalign=1, hexpand=True)
        label.get_style_context().add_class("dim-label")
        top.pack_start(label, True, True, 0)
        self.pack_start(top, False, False, 0)
        scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        self.body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=3)
        scroll.add(self.body); self.pack_start(scroll, True, True, 0)
        self.render()

    @staticmethod
    def label(text, small=False):
        label = Gtk.Label(label=text, xalign=0, wrap=True)
        label.set_max_width_chars(65)
        if small:
            label.get_style_context().add_class("dim-label")
        return label

    def update(self, data):
        self.data = data
        self.render()

    def render(self):
        for child in self.body.get_children():
            child.destroy()
        if self.data is None:
            self.body.pack_start(self.label("Reading your score history… (it appears here as soon as your plays are read and analysed; progress is shown at the bottom)"), False, False, 0)
            self.body.show_all()
            return
        pages = self.data["pages"]
        selected = self.keys.get_active_id()
        if selected == "all":
            self.body.pack_start(self.label("Your keymodes"), False, False, 0)
            self.body.pack_start(self.label("Each rating uses its own keymode's difficulty scale. There is no cross-keymode average.", True), False, False, 0)
            for k in range(4, 11):
                page = pages.get(k, pages.get(str(k), {}))
                r = page.get("groups", {}).get("overall", {})
                value = f"{r['value']:.1f}" if r.get("value") is not None else "—"
                button = Gtk.Button(relief=Gtk.ReliefStyle.NONE)
                button.add(self.label(f"{k}K     {value}    ·    {page.get('maps', 0):,} charts / {page.get('plays', 0):,} plays"))
                button.connect("clicked", lambda _b, key=k: self.keys.set_active_id(str(key)))
                self.body.pack_start(button, False, False, 0)
        else:
            keys = int(selected or 7)
            page = pages.get(keys, pages.get(str(keys), {}))
            groups = page.get("groups", {})
            overall = groups.get("overall", {})
            value = overall.get("value")
            title = Gtk.Label(xalign=0)
            dan = dans.ability(keys, value)
            title.set_markup(f"<span size='220%' weight='bold'>{value:.1f}</span>  <b>your {keys}K level</b>"
                             + (f"  ·  ≈ {GLib.markup_escape_text(dan)}" if dan else "")
                             if value is not None else f"<b>{keys}K · not enough history yet</b>")
            self.body.pack_start(title, False, False, 0)
            peak, session = page.get("peak"), page.get("session")
            lines = []
            if value is not None and peak and peak[1] > value + .05:
                pdan = dans.ability(keys, peak[1])
                lines.append(f"Peak {peak[1]:.1f} in {month_name(peak[0])}" + (f" (≈ {pdan})" if pdan else ""))
            elif value is not None and peak:
                lines.append("This is your peak so far.")
            current = overall.get("current")
            if value is not None and session and current is not None:
                lines.append(f"This session: {current:.1f} (○ on the bars)")
            elif value is not None:
                lines.append("Not warmed up: the first maps of a session play lower than this, so no session "
                             "number is shown until you have played a few.")
            if lines:
                self.body.pack_start(self.label("\n".join(lines)), False, False, 0)
            if len(page.get("history", ())) >= 2:
                self.body.pack_start(HistoryChart(keys, page["history"], peak or page["history"][-1]), False, False, 0)
            self.body.pack_start(self.label(
                f"{page.get('maps', 0):,} charts · {page.get('plays', 0):,} plays. Level = the chart difficulty you "
                "play at ~94% when warmed up, from your recent months. ≈ dan: the dan whose songs sit there "
                "(one song, not passing the course).", True), False, False, 0)
            shown = [r for r in groups.values() if r.get("value") is not None]
            scale = Scale(min([r["low"] for r in shown] + ([peak[1]] if peak else []), default=1.) - .2,
                          max([r["high"] for r in shown] + ([peak[1]] if peak else []), default=10.) + .2)
            self.scale, self.session = scale, session
            # Every row's bar starts and ends at the same x, under the dan axis.
            self.names = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL)
            self.tails = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL)
            legend = Gtk.Grid(column_spacing=8)
            legend.attach(self.label("● level   band: uncertainty" + ("   ○ this session" if session else ""), True), 0, 0, 1, 1)
            self.body.pack_start(legend, False, False, 0)
            axis = Gtk.Grid(column_spacing=8)
            spacer, tail = Gtk.Label(), Gtk.Label()
            self.names.add_widget(spacer); self.tails.add_widget(tail)
            axis.attach(spacer, 0, 0, 1, 1)
            axis.attach(DanAxis(keys, scale), 1, 0, 1, 1)
            axis.attach(tail, 2, 0, 1, 1)
            self.body.pack_start(axis, False, False, 0)
            parents = skill_practice.tree(keys)
            self.expanders = {}
            for parent in parents:
                if parent.children:
                    identity = (keys, parent.key)
                    children = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
                    for child in parent.children:
                        children.pack_start(self._skill_row(keys, child, groups, indent=True), False, False, 0)
                    reveal=Gtk.Revealer(reveal_child=identity in self.expanded)
                    reveal.add(children)
                    toggle=Gtk.ToggleButton(relief=Gtk.ReliefStyle.NONE,active=identity in self.expanded)
                    toggle.set_label(("▾ " if toggle.get_active() else "▸ ")+parent.name)
                    toggle.get_child().set_xalign(0)
                    toggle.set_tooltip_text(parent.tip or "Expand this family into its subskills.")
                    def toggled(widget, ident=identity, panel=reveal, name=parent.name):
                        active=widget.get_active()
                        if active:self.expanded.add(ident)
                        else:self.expanded.discard(ident)
                        widget.set_label(("▾ " if active else "▸ ")+name)
                        panel.set_reveal_child(active)
                    toggle.connect("toggled",toggled)
                    self.expanders[parent.key] = toggle
                    row=self._skill_row(keys,parent,groups,toggle)
                    row.pack_start(reveal,False,False,0)
                    self.body.pack_start(row,False,False,0)
                else:
                    self.body.pack_start(self._skill_row(keys, parent, groups), False, False, 0)
            reliable = [(p.key, groups[p.key]) for p in parents if p.key in groups and
                        groups[p.key].get("value") is not None and groups[p.key]["maps"] >= 10]
            if reliable and value is not None:
                high = max(reliable, key=lambda p: p[1]["value"])
                low = min(reliable, key=lambda p: p[1]["value"])
                if high[1]["low"] > low[1]["high"]:
                    self.body.pack_start(self.label(f"Profile contrast: {skill_practice.NAMES[high[0]]} is better supported at higher difficulty than {skill_practice.NAMES[low[0]]}."), False, False, 0)
            diagnostic = Gtk.Expander(label="Prediction strengths & blind spots")
            messages = [f"{skill_practice.NAMES[d['skill']]}: {abs(d['delta']):.1f} accuracy points "
                        f"{'above' if d['delta'] > 0 else 'below'} the saved pre-play predictions across {d['maps']} charts."
                        for d in page.get("diagnostics", ())]
            messages.append("These are model diagnostics, not proof of a personal weakness. Mixed patterns, session form and older predictor versions may contribute.")
            if len(messages) == 1:
                messages.insert(0, "No repeatable large prediction bias supported by enough distinct tracked charts yet.")
            diagnostic.add(self.label("\n\n".join(messages), True))
            self.body.pack_start(diagnostic, False, False, 0)
            mash = page.get("special", {}).get("mash_maps", 0)
            if mash:
                self.body.pack_start(self.label(f"Special-pattern evidence: Mash appears materially in {mash} charts. It is an execution/difficulty correction, not a separate practice goal.", True), False, False, 0)
        self.body.show_all()

    def _skill_row(self, keys, node, groups, toggle=None, indent=False):
        row = groups.get(node.key, {"value": None, "maps": 0, "confidence": "Not enough evidence"})
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        grid = Gtk.Grid(column_spacing=8, row_spacing=2, hexpand=True)
        # A parent summary remains visible while its children expand below it.
        name = toggle or self.label(node.name)
        name.set_size_request(105, -1)
        name.set_margin_start(22 if indent else 0)      # the name indents, the bar stays on the shared axis
        name.set_tooltip_text(node.tip)
        self.names.add_widget(name)
        grid.attach(name, 0, 0, 1, 1)
        bar = SkillBar(keys, row, self.scale, self.session)
        grid.attach(bar, 1, 0, 1, 1)
        dan = dans.ability(keys, row.get("value"), node.key)
        value = Gtk.Label(xalign=0)
        value.set_size_request(130, -1)
        value.set_markup("—" if row.get("value") is None else
                         f"<b>{row['value']:.1f}</b>" + (f"  <small>≈ {GLib.markup_escape_text(dan)}</small>" if dan else ""))
        practice = Gtk.Button(label="Practice", relief=Gtk.ReliefStyle.NONE)
        practice.set_tooltip_text(f"Practice {node.name} in {keys}K")
        practice.connect("clicked", lambda *_: self.practice(keys, node.key))
        tail = Gtk.Box(spacing=8)
        tail.pack_start(value, False, False, 0); tail.pack_end(practice, False, False, 0)
        self.tails.add_widget(tail)
        grid.attach(tail, 2, 0, 1, 1)
        session = f" · this session {row['current']:.1f}" if self.session and row.get("current") is not None and row.get("value") is not None else ""
        grid.attach(self.label(f"{row['confidence']} · {row['maps']:,} charts{session}", True), 1, 1, 2, 1)
        if row.get("value") is not None:
            examples = "\n".join(f"{v['title']} · {v['rate']:.2f}× · difficulty {v['difficulty']:.1f}" for v in row.get("examples", ()))
            bar.set_tooltip_text(f"Level {row['value']:.2f}; likely between {row['low']:.2f} and {row['high']:.2f}. "
                                 f"{row['plays']} plays on {row['maps']} charts." + ("\n"+examples if examples else ""))
        elif row.get("extrapolated"):
            bar.set_tooltip_text(f"Evidence reaches difficulty {row['tested_to']:.1f}; the fitted limit is not measured.")
        box.pack_start(grid, False, False, 0)
        return box

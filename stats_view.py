"""Native GTK player profile. Rendering only; data comes from player_stats."""
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

import skill_practice


class AbilityBar(Gtk.DrawingArea):
    def __init__(self, row, maximum):
        super().__init__(hexpand=True, valign=Gtk.Align.CENTER)
        self.row, self.maximum = row, max(1., maximum)
        self.set_size_request(100, 25)
        self.connect("draw", self.draw)

    def draw(self, widget, cr):
        width, height = self.get_allocated_width(), self.get_allocated_height()
        ink = self.get_style_context().get_color(Gtk.StateFlags.NORMAL)
        x = lambda value: 3+(width-6)*min(1., max(0., value)/self.maximum)
        y = height/2
        cr.set_source_rgba(ink.red, ink.green, ink.blue, .12)
        cr.set_line_width(5); cr.move_to(3, y); cr.line_to(width-3, y); cr.stroke()
        r = self.row
        if r.get("value") is None:
            return False
        accent = self.get_style_context().lookup_color("theme_selected_bg_color")
        color = accent[1] if accent[0] else ink
        cr.set_source_rgba(color.red, color.green, color.blue, .72)
        cr.set_line_width(5); cr.move_to(3, y); cr.line_to(x(r["value"]), y); cr.stroke()
        cr.set_source_rgba(ink.red, ink.green, ink.blue, .45)
        cr.set_line_width(1); cr.move_to(x(r["low"]), y); cr.line_to(x(r["high"]), y); cr.stroke()
        for value in (r["low"], r["high"]):
            cr.move_to(x(value), y-5); cr.line_to(x(value), y+5); cr.stroke()
        # A diamond is current form; a hollow diamond means not activated yet.
        if r.get("current") is not None:
            at = x(r["current"])
            cr.set_source_rgba(ink.red, ink.green, ink.blue, .95)
            cr.move_to(at, y-6); cr.line_to(at+5, y); cr.line_to(at, y+6); cr.line_to(at-5, y); cr.close_path()
            if r.get("activation", 0) >= 1.5:
                cr.fill()
            else:
                cr.set_line_width(1.5); cr.stroke()
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
            self.body.pack_start(self.label("Open Stats to read your existing score history. No chart scan is required."), False, False, 0)
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
            title.set_markup(f"<span size='220%' weight='bold'>{value:.1f}</span>  <b>{keys}K overall</b>" if value is not None else f"<b>{keys}K · not enough history yet</b>")
            self.body.pack_start(title, False, False, 0)
            self.body.pack_start(self.label(f"{page.get('maps', 0):,} unique charts · {page.get('plays', 0):,} usable plays\n"
                                           "Estimated chart difficulty at ~94% displayed accuracy, conditioned on each skill.", True), False, False, 0)
            self.body.pack_start(self.label("Bar: long-term ability   ◆: current session   ◇: not activated\nWhisker: estimate spread, not a calibrated confidence interval.", True), False, False, 0)
            maximum = max((r.get("high", 0) for r in groups.values()), default=1)*1.08
            parents = skill_practice.tree(keys)
            self.body.pack_start(self.label("Expand a family for its subskills. Practice uses the same selections as the Skills tab.", True), False, False, 0)
            self.expanders = {}
            for parent in parents:
                if parent.children:
                    identity = (keys, parent.key)
                    children = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin_start=22)
                    for child in parent.children:
                        children.pack_start(self._skill_row(keys, child, groups, maximum), False, False, 0)
                    reveal=Gtk.Revealer(reveal_child=identity in self.expanded)
                    reveal.add(children)
                    toggle=Gtk.ToggleButton(relief=Gtk.ReliefStyle.NONE,active=identity in self.expanded)
                    toggle.set_label(("▾ " if toggle.get_active() else "▸ ")+parent.name)
                    toggle.set_tooltip_text(parent.tip or "Expand this family into its subskills.")
                    def toggled(widget, ident=identity, panel=reveal, name=parent.name):
                        active=widget.get_active()
                        if active:self.expanded.add(ident)
                        else:self.expanded.discard(ident)
                        widget.set_label(("▾ " if active else "▸ ")+name)
                        panel.set_reveal_child(active)
                    toggle.connect("toggled",toggled)
                    self.expanders[parent.key] = toggle
                    row=self._skill_row(keys,parent,groups,maximum,toggle)
                    row.pack_start(reveal,False,False,0)
                    self.body.pack_start(row,False,False,0)
                else:
                    self.body.pack_start(self._skill_row(keys, parent, groups, maximum), False, False, 0)
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

    def _skill_row(self, keys, node, groups, maximum, toggle=None):
        row = groups.get(node.key, {"value": None, "maps": 0, "confidence": "Not enough evidence"})
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        grid = Gtk.Grid(column_spacing=8, row_spacing=2, hexpand=True)
        # A parent summary remains visible while its children expand below it.
        name = toggle or self.label(node.name)
        name.set_size_request(105, -1)
        name.set_tooltip_text(node.tip)
        grid.attach(name, 0, 0, 1, 1)
        bar = AbilityBar(row, maximum)
        grid.attach(bar, 1, 0, 1, 1)
        value = "—" if row.get("value") is None else f"{row['value']:.1f} → {row.get('current', row['value']):.1f}"
        grid.attach(Gtk.Label(label=value, xalign=1), 2, 0, 1, 1)
        practice = Gtk.Button(label="Practice", relief=Gtk.ReliefStyle.NONE)
        practice.set_tooltip_text(f"Practice {node.name} in {keys}K")
        practice.connect("clicked", lambda *_: self.practice(keys, node.key))
        grid.attach(practice, 3, 0, 1, 1)
        grid.attach(self.label(f"{row['confidence']} · {row['maps']:,} charts", True), 1, 1, 3, 1)
        if row.get("value") is not None:
            examples = "\n".join(f"{v['title']} · {v['rate']:.2f}× · difficulty {v['difficulty']:.1f}" for v in row.get("examples", ()))
            bar.set_tooltip_text(f"Long-term {row['value']:.2f}; estimate spread {row['low']:.2f}–{row['high']:.2f}. "
                                 f"{row['plays']} plays on {row['maps']} independent charts." + ("\n"+examples if examples else ""))
        elif row.get("extrapolated"):
            bar.set_tooltip_text(f"Evidence reaches difficulty {row['tested_to']:.1f}; the fitted limit is not measured.")
        box.pack_start(grid, False, False, 0)
        return box

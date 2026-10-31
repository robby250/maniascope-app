"""Skills chips on the Skills tab's settings page, over the same taxonomy as player Stats.
Replaced a popover of checkboxes whose third state read as dots (user 2026-10-31: "kinda scuffed")."""
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk
import skill_practice as skills

GREEN, RED = "#57c26e", "#e5534b"
OFTEN = (.25, .5, 1., 2., 4.)       # how-often slider: multiplies a skill's share of the list
OFTEN_NAMES = dict(zip(OFTEN, ("Rare", "Less", "Normal", "More", "Often")))


class Chip(Gtk.Button):
    """One skill. value: 1 included, -1 excluded, 0 off; mixed: its subskills differ."""
    def __init__(self, node, clicked):
        super().__init__(label=node.name, can_focus=True)
        self.node, self.value, self.mixed = node, 0, False
        self.get_style_context().add_class("skill-chip")
        self.set_tooltip_text(node.tip or "Broad practice for this family; ▾ opens its subskills.")
        self.connect("clicked", lambda _b: clicked(node.key))

    def show_state(self, value, mixed):
        self.value, self.mixed = value, mixed
        mark = "–" if mixed else {1: f'<span foreground="{GREEN}">✓</span>',
                                  -1: f'<span foreground="{RED}"><b>✕</b></span>'}.get(value)
        name = GLib.markup_escape_text(self.node.name)
        self.get_child().set_markup(f"{mark}  {name}" if mark else name)
        style = self.get_style_context()
        for cls, on in (("skill-on", value == 1 and not mixed), ("skill-off", value == -1 and not mixed)):
            (style.add_class if on else style.remove_class)(cls)
        word = "mixed" if mixed else {1: "included", -1: "excluded"}.get(value, "off")
        self.get_accessible().set_name(f"{self.node.name}: {word}")


class SkillPicker(Gtk.Box):
    def __init__(self, keys, selected, changed, weights, weight_changed, slider):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.changed, self.selected = changed, skills.normalize(selected)
        self.weights, self.weight_changed, self.slider = weights, weight_changed, slider
        self.expanded, self.checks, self.all_checks, self.reveals = set(), {}, {}, {}
        self.any_button = Gtk.Button(label="Any · regular practice", halign=Gtk.Align.START)
        self.any_button.connect("clicked", lambda *_: self.changed(()))
        self.pack_start(self.any_button, False, False, 0)
        hint = Gtk.Label(label="Click: include → exclude → off. ▾ opens subskills. A map needs one included skill.", xalign=0)
        hint.get_style_context().add_class("dim-label")
        self.pack_start(hint, False, False, 0)
        self.body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.pack_start(self.body, False, False, 0)
        self.often = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.pack_start(self.often, False, False, 0)
        self._often_keys = None
        self.set_keys(keys)

    def set_keys(self, keys):
        self.keys = tuple(keys)
        for child in self.body.get_children():
            self.body.remove(child)
        self.checks, self.all_checks = {}, {}
        modes = ([4] if 4 in keys else []) + ([7] if any(k >= 5 for k in keys) else [])
        for mode in modes:
            heading = Gtk.Label(xalign=0, margin_top=4)
            heading.set_markup("<b>4K</b>" if mode == 4 else "<b>5K–10K</b>")
            self.body.pack_start(heading, False, False, 0)
            flow = Gtk.FlowBox(selection_mode=Gtk.SelectionMode.NONE, homogeneous=False, column_spacing=4,
                               row_spacing=4, max_children_per_line=8)
            self.body.pack_start(flow, False, False, 0)
            for parent in skills.tree(mode):
                flow.add(self._chip_with_arrow(mode, parent) if parent.children else self._chip(parent))
            for child in flow.get_children():
                child.set_can_focus(False)      # the chip inside takes focus, not its flow cell
            for parent in skills.tree(mode):
                if parent.children:
                    self.body.pack_start(self._children(mode, parent), False, False, 0)
        if not modes:
            self.body.pack_start(Gtk.Label(label="Select a keymode first."), False, False, 0)
        self.body.show_all()
        self.set_selection(self.selected)

    def _chip(self, node):
        chip = Chip(node, self._click)
        self.checks.setdefault(node.key, chip)
        self.all_checks.setdefault(node.key, []).append(chip)
        return chip

    def _chip_with_arrow(self, mode, parent):
        pair = Gtk.Box()
        pair.get_style_context().add_class("linked")
        pair.pack_start(self._chip(parent), False, False, 0)
        arrow = Gtk.ToggleButton(label="▾", active=(mode, parent.key) in self.expanded)
        arrow.set_tooltip_text("Subskills")
        arrow.get_accessible().set_name(f"{parent.name} subskills")
        arrow.connect("toggled", self._expand, (mode, parent.key))
        pair.pack_start(arrow, False, False, 0)
        pair.arrow = arrow
        return pair

    def _children(self, mode, parent):
        row = Gtk.Box(spacing=4, margin_start=12)
        name = Gtk.Label(label=parent.name, xalign=0)
        name.get_style_context().add_class("dim-label")
        row.pack_start(name, False, False, 0)
        for child in parent.children:
            row.pack_start(self._chip(child), False, False, 0)
        reveal = Gtk.Revealer(transition_type=Gtk.RevealerTransitionType.NONE,
                              reveal_child=(mode, parent.key) in self.expanded)
        reveal.add(row)
        self.reveals[(mode, parent.key)] = reveal
        return reveal

    def _expand(self, arrow, ident):
        if arrow.get_active():
            self.expanded.add(ident)
        else:
            self.expanded.discard(ident)
        self.reveals[ident].set_reveal_child(arrow.get_active())

    def _click(self, key):
        selected = skills.cycle(self.selected, key)
        self.set_selection(selected)    # reflect the click before catalogue work and callbacks
        self.changed(selected)

    def set_selection(self, selected):
        self.selected = skills.normalize(selected)
        for key, chips in self.all_checks.items():
            for chip in chips:
                chip.show_state(skills.state(self.selected, key), skills.mixed(self.selected, key))
        self._often_rows()

    def _often_rows(self):
        """A how-often slider per included skill, once there are two to share the list."""
        included = tuple(k for k in self.selected if not k.startswith("!"))
        included = included if len(included) > 1 else ()
        if included == self._often_keys:
            return
        self._often_keys = included
        for row in self.often.get_children():
            self.often.remove(row)
        for key in included:
            self.often.pack_start(self.slider(skills.NAMES[key], self.weights.get(key, 1.), OFTEN_NAMES.get,
                                              lambda v, k=key: self.weight_changed(k, v), values=OFTEN,
                                              tooltip="How often this skill comes up against the other included ones."),
                                  False, False, 0)
        self.often.show_all()

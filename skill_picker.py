"""Expandable GTK selector backed by the same taxonomy as player Stats."""
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Pango
import skill_practice as skills


class SkillPicker(Gtk.MenuButton):
    def __init__(self, keys, selected, changed):
        super().__init__()
        self.changed, self.selected = changed, skills.normalize(selected)
        self.expanded, self.checks, self.all_checks = set(), {}, {}
        self.updating = False
        self.caption = Gtk.Label(ellipsize=Pango.EllipsizeMode.END, max_width_chars=36)
        self.add(self.caption)
        self.set_keys(keys)

    def set_keys(self, keys):
        self.keys = tuple(keys)
        pop = Gtk.Popover()
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=10)
        self.any_button = Gtk.Button(label="Any · regular practice")
        self.any_button.connect("clicked", lambda *_: self.changed(()))
        root.pack_start(self.any_button, False, False, 0)
        hint = Gtk.Label(label="Click: include → exclude → neutral. Arrow: expand.\nA parent applies to its family; included skills mean any.", xalign=0)
        hint.get_style_context().add_class("dim-label")
        root.pack_start(hint, False, False, 0)
        scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER,
                                   max_content_height=470, propagate_natural_height=True)
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        scroll.add(body); root.pack_start(scroll, True, True, 0)
        self.checks, self.all_checks = {}, {}
        self.arrow_width = arrow_width = Gtk.SizeGroup(mode=Gtk.SizeGroupMode.HORIZONTAL)
        modes = ([4] if 4 in keys else []) + ([7] if any(k >= 5 for k in keys) else [])
        for mode in modes:
            heading = Gtk.Label(xalign=0, margin_top=5)
            heading.set_markup("<b>4K</b>" if mode == 4 else "<b>5K–10K</b>")
            body.pack_start(heading, False, False, 0)
            for parent in skills.tree(mode):
                check = self._check(parent)
                if not parent.children:
                    header = Gtk.Box(spacing=3)
                    spacer = Gtk.Box()
                    arrow_width.add_widget(spacer)
                    header.pack_start(spacer, False, False, 0)
                    header.pack_start(check, True, True, 0)
                    body.pack_start(header, False, False, 0)
                    continue
                group = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
                header = Gtk.Box(spacing=3)
                arrow = Gtk.ToggleButton(relief=Gtk.ReliefStyle.NONE)
                arrow_width.add_widget(arrow)
                arrow.set_tooltip_text("Expand/collapse subskills without changing selection")
                icon = Gtk.Image()
                arrow.add(icon)
                header.pack_start(arrow, False, False, 0)
                header.pack_start(check, True, True, 0)
                group.pack_start(header, False, False, 0)
                children = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, margin_start=24)
                for child in parent.children:
                    children.pack_start(self._check(child), False, False, 0)
                reveal = Gtk.Revealer(transition_type=Gtk.RevealerTransitionType.NONE)
                reveal.add(children); group.pack_start(reveal, False, False, 0)
                def expanded(button, ident=(mode, parent.key), reveal=reveal, icon=icon):
                    active = button.get_active()
                    reveal.set_reveal_child(active)
                    icon.set_from_icon_name("pan-down-symbolic" if active else "pan-end-symbolic", Gtk.IconSize.MENU)
                    if active: self.expanded.add(ident)
                    else: self.expanded.discard(ident)
                arrow.connect("toggled", expanded)
                arrow.set_active((mode, parent.key) in self.expanded)
                expanded(arrow)
                body.pack_start(group, False, False, 0)
        if not modes:
            body.pack_start(Gtk.Label(label="Select a keymode first."), False, False, 0)
        pop.add(root); root.show_all()
        old = self.get_popover()
        self.set_popover(pop)
        if old: old.destroy()
        self.set_selection(self.selected)

    def _check(self, node):
        check = Gtk.CheckButton(label=node.name)
        check.set_tooltip_text(node.tip or "Broad practice for this family; expand to choose a subskill.")
        self.checks.setdefault(node.key, check)
        self.all_checks.setdefault(node.key, []).append(check)
        def activate(*_args):
            if not self.updating:
                selected = skills.cycle(self.selected, node.key)
                # Reflect the click immediately, before catalogue work/callbacks.
                self.set_selection(selected)
                self.changed(selected)
            return True
        # Native button activation handles label/indicator clicks, dragging,
        # keyboard and accessibility. Its binary state is replaced immediately
        # from our signed selection, never inferred from get_active(). The
        # updating guard prevents set_active() from recursively cycling it.
        check.connect("clicked", activate)
        return check

    def set_selection(self, selected):
        self.selected = skills.normalize(selected)
        self.updating = True
        for key, checks in self.all_checks.items():
            for check in checks:
                value = skills.state(self.selected, key)
                check.set_active(value == 1)
                check.set_inconsistent(value == -1 or skills.mixed(self.selected, key))
                check.set_label(skills.NAMES[key] + ("  · excluded" if value == -1 else ""))
        self.updating = False
        self.caption.set_text(skills.label(self.selected))
        hidden = [k for k in self.selected if k.lstrip("!") not in self.all_checks]
        self.set_tooltip_text(skills.label(self.selected) +
            (" · Some selections belong to other keymodes; they are retained, not converted to Any." if hidden else
             " · Whole-map suitability; no forced subskill rotation."))

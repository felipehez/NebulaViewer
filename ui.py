"""Minimal docked UI for NebulaViewer drawn with pyglet shapes + labels (no extra dependencies).

Panel (right): collapsible sections of sliders / toggles / choices / buttons, scrollable with the wheel.
  slider: click or drag sets the value; Shift-drag = fine; right click = default value.
  choice: left click = next, right click = previous.
TimelineBar (bottom): transport buttons + track with the cached snapshots; click/drag to scrub.
"""
import math
import pyglet
from pyglet import shapes

FONT = "Segoe UI"
MONO = "Consolas"
C_BG = (22, 24, 30, 235)
C_SEC = (38, 42, 54, 255)
C_TRACK = (48, 52, 64, 255)
C_FILL = (70, 120, 190, 255)
C_FILL_PEND = (190, 130, 60, 255)
C_HOVER = (60, 66, 82, 255)
C_TEXT = (225, 228, 235, 255)
C_DIM = (150, 156, 170, 255)
C_ON = (80, 170, 110, 255)
C_REC = (210, 50, 50, 255)
PANEL_W = 330


class Widget:
    h = 22
    help = ""
    section = None

    def __init__(self, label, help=""):
        self.label = label
        self.help = help
        self.items = []          # pyglet drawables
        self.x = self.y = self.w = 0

    def visible(self):
        return self.section is None or not self.section.collapsed

    def build(self, batch, groups):
        pass

    def set_visible(self, v):
        for it in self.items:
            it.visible = v

    def place(self, x, y, w):
        self.x, self.y, self.w = x, y, w

    def refresh(self):
        pass

    def hit(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def press(self, x, y, button, mods):
        pass

    def drag(self, x, y, dx, dy, mods):
        pass

    def release(self):
        pass


class Section(Widget):
    h = 24

    def __init__(self, label, collapsed=False, help=""):
        super().__init__(label, help)
        self.collapsed = collapsed
        self.section = None

    def build(self, batch, g):
        self.bg = shapes.Rectangle(0, 0, 10, self.h - 2, color=C_SEC, batch=batch, group=g[0])
        self.lbl = pyglet.text.Label("", font_name=FONT, font_size=10, weight="bold", color=C_TEXT,
                                     anchor_y="center", batch=batch, group=g[2])
        self.items = [self.bg, self.lbl]

    def place(self, x, y, w):
        super().place(x, y, w)
        self.bg.position = (x, y + 1)
        self.bg.width = w
        self.lbl.x, self.lbl.y = x + 8, y + self.h / 2
        self.refresh()

    def refresh(self):
        t = ("▸ " if self.collapsed else "▾ ") + self.label
        if self.lbl.text != t:
            self.lbl.text = t

    def press(self, x, y, button, mods):
        self.collapsed = not self.collapsed
        return "layout"


class Slider(Widget):
    def __init__(self, label, get, set, lo, hi, log=False, fmt="{:.3g}", default=None, integer=False,
                 help="", pending=None, on_release=None):
        super().__init__(label, help)
        self.get, self.set_, self.lo, self.hi, self.log = get, set, lo, hi, log
        self.fmt, self.default, self.integer = fmt, default, integer
        self.pending = pending          # callable -> True when the value differs from the running sim
        self.on_release = on_release
        self._last = None
        self._drag = False

    def _norm(self, v):
        if self.log:
            v = max(v, self.lo)
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def _denorm(self, u):
        u = min(max(u, 0.0), 1.0)
        if self.log:
            v = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        else:
            v = self.lo + u * (self.hi - self.lo)
        return int(round(v)) if self.integer else v

    def build(self, batch, g):
        self.track = shapes.Rectangle(0, 0, 10, self.h - 4, color=C_TRACK, batch=batch, group=g[0])
        self.fill = shapes.Rectangle(0, 0, 1, self.h - 4, color=C_FILL, batch=batch, group=g[1])
        self.lbl = pyglet.text.Label(self.label, font_name=FONT, font_size=9, color=C_TEXT, anchor_y="center",
                                     batch=batch, group=g[2])
        self.val = pyglet.text.Label("", font_name=MONO, font_size=9, color=C_TEXT, anchor_x="right",
                                     anchor_y="center", batch=batch, group=g[2])
        self.items = [self.track, self.fill, self.lbl, self.val]

    def place(self, x, y, w):
        super().place(x, y, w)
        self.track.position = (x, y + 2)
        self.track.width = w
        self.fill.position = (x, y + 2)
        self.lbl.x, self.lbl.y = x + 6, y + self.h / 2
        self.val.x, self.val.y = x + w - 6, y + self.h / 2
        self._last = None
        self.refresh()

    def refresh(self):
        v = self.get()
        pend = bool(self.pending and self.pending())
        key = (v, pend)
        if key == self._last:
            return
        self._last = key
        self.fill.width = max(1, int(self.w * min(max(self._norm(v), 0.0), 1.0)))
        self.fill.color = C_FILL_PEND if pend else C_FILL
        self.val.text = (self.fmt.format(v)) + (" *" if pend else "")

    def _set_from_x(self, x):
        self.set_(self._denorm((x - self.x) / max(self.w, 1)))

    def press(self, x, y, button, mods):
        if button == pyglet.window.mouse.RIGHT:
            if self.default is not None:
                self.set_(self.default)
                if self.on_release:
                    self.on_release()
            return None
        self._drag = True
        self._fine_u = self._norm(self.get())
        if not (mods & pyglet.window.key.MOD_SHIFT):
            self._set_from_x(x)
        return "capture"

    def drag(self, x, y, dx, dy, mods):
        if not self._drag:
            return
        if mods & pyglet.window.key.MOD_SHIFT:
            self._fine_u += 0.1 * dx / max(self.w, 1)
            self.set_(self._denorm(self._fine_u))
        else:
            self._fine_u = (x - self.x) / max(self.w, 1)
            self._set_from_x(x)

    def release(self):
        if self._drag and self.on_release:
            self.on_release()
        self._drag = False


class Toggle(Widget):
    def __init__(self, label, get, set, help="", on_color=C_ON):
        super().__init__(label, help)
        self.get, self.set_ = get, set
        self.on_color = on_color
        self._last = None

    def build(self, batch, g):
        self.bg = shapes.Rectangle(0, 0, 10, self.h - 4, color=C_TRACK, batch=batch, group=g[0])
        self.box = shapes.Rectangle(0, 0, 12, 12, color=C_DIM, batch=batch, group=g[1])
        self.lbl = pyglet.text.Label(self.label, font_name=FONT, font_size=9, color=C_TEXT, anchor_y="center",
                                     batch=batch, group=g[2])
        self.items = [self.bg, self.box, self.lbl]

    def place(self, x, y, w):
        super().place(x, y, w)
        self.bg.position = (x, y + 2)
        self.bg.width = w
        self.box.position = (x + w - 20, y + self.h / 2 - 6)
        self.lbl.x, self.lbl.y = x + 6, y + self.h / 2
        self._last = None
        self.refresh()

    def refresh(self):
        v = bool(self.get())
        if v != self._last:
            self._last = v
            self.box.color = self.on_color if v else (70, 74, 86, 255)

    def press(self, x, y, button, mods):
        self.set_(not bool(self.get()))
        return None


class Choice(Widget):
    def __init__(self, label, options, get, set, help="", pending=None):
        super().__init__(label, help)
        self.options, self.get, self.set_ = options, get, set
        self.pending = pending
        self._last = None

    def build(self, batch, g):
        self.bg = shapes.Rectangle(0, 0, 10, self.h - 4, color=C_TRACK, batch=batch, group=g[0])
        self.lbl = pyglet.text.Label(self.label, font_name=FONT, font_size=9, color=C_TEXT, anchor_y="center",
                                     batch=batch, group=g[2])
        self.val = pyglet.text.Label("", font_name=MONO, font_size=9, color=(140, 190, 255, 255), anchor_x="right",
                                     anchor_y="center", batch=batch, group=g[2])
        self.items = [self.bg, self.lbl, self.val]

    def place(self, x, y, w):
        super().place(x, y, w)
        self.bg.position = (x, y + 2)
        self.bg.width = w
        self.lbl.x, self.lbl.y = x + 6, y + self.h / 2
        self.val.x, self.val.y = x + w - 6, y + self.h / 2
        self._last = None
        self.refresh()

    def refresh(self):
        v = self.get()
        pend = bool(self.pending and self.pending())
        if (v, pend) != self._last:
            self._last = (v, pend)
            self.val.text = f"‹ {v} ›" + (" *" if pend else "")
            self.val.color = C_FILL_PEND if pend else (140, 190, 255, 255)

    def press(self, x, y, button, mods):
        opts = self.options() if callable(self.options) else self.options
        cur = self.get()
        i = opts.index(cur) if cur in opts else 0
        i = (i + (-1 if button == pyglet.window.mouse.RIGHT else 1)) % len(opts)
        self.set_(opts[i])
        return "layout"


class Button(Widget):
    h = 26

    def __init__(self, label, action, help="", color=None, label_fn=None, color_fn=None):
        super().__init__(label, help)
        self.action = action
        self.color = color or (58, 92, 140, 255)
        self.label_fn, self.color_fn = label_fn, color_fn
        self._last = None

    def build(self, batch, g):
        self.bg = shapes.Rectangle(0, 0, 10, self.h - 4, color=self.color, batch=batch, group=g[0])
        self.lbl = pyglet.text.Label(self.label, font_name=FONT, font_size=9, weight="bold", color=C_TEXT,
                                     anchor_x="center", anchor_y="center", batch=batch, group=g[2])
        self.items = [self.bg, self.lbl]

    def place(self, x, y, w):
        super().place(x, y, w)
        self.bg.position = (x, y + 2)
        self.bg.width = w
        self.lbl.x, self.lbl.y = x + w / 2, y + self.h / 2
        self._last = None
        self.refresh()

    def refresh(self):
        t = self.label_fn() if self.label_fn else self.label
        c = self.color_fn() if self.color_fn else self.color
        if (t, c) != self._last:
            self._last = (t, c)
            self.lbl.text = t
            self.bg.color = c

    def press(self, x, y, button, mods):
        self.action()
        return "layout"


class Row(Widget):
    """several buttons side by side"""
    h = 26

    def __init__(self, *buttons):
        super().__init__("")
        self.buttons = list(buttons)

    def build(self, batch, g):
        self.items = []
        for b in self.buttons:
            b.build(batch, g)
            self.items += b.items

    def place(self, x, y, w):
        super().place(x, y, w)
        n = len(self.buttons)
        bw = (w - 4 * (n - 1)) / n
        for i, b in enumerate(self.buttons):
            b.place(x + i * (bw + 4), y, bw)

    def refresh(self):
        for b in self.buttons:
            b.refresh()

    def press(self, x, y, button, mods):
        for b in self.buttons:
            if b.hit(x, y):
                self.help = b.help
                return b.press(x, y, button, mods)


class Info(Widget):
    h = 18

    def __init__(self, text_fn):
        super().__init__("")
        self.text_fn = text_fn
        self._last = None

    def build(self, batch, g):
        self.lbl = pyglet.text.Label("", font_name=MONO, font_size=8, color=C_DIM, anchor_y="center",
                                     batch=batch, group=g[2])
        self.items = [self.lbl]

    def place(self, x, y, w):
        super().place(x, y, w)
        self.lbl.x, self.lbl.y = x + 4, y + self.h / 2
        self._last = None
        self.refresh()

    def refresh(self):
        t = self.text_fn()
        if t != self._last:
            self._last = t
            self.lbl.text = t


class Panel:
    """docked on the right side of the window"""

    def __init__(self, width=PANEL_W):
        self.width = width
        self.widgets = []
        self.batch = pyglet.graphics.Batch()
        self.groups = [pyglet.graphics.Group(order=1), pyglet.graphics.Group(order=2),
                       pyglet.graphics.Group(order=3)]
        self.bg = shapes.Rectangle(0, 0, width, 10, color=C_BG, batch=self.batch,
                                   group=pyglet.graphics.Group(order=0))
        self.help_bg = shapes.Rectangle(0, 0, width, 44, color=(14, 15, 20, 250), batch=self.batch,
                                        group=pyglet.graphics.Group(order=4))
        self.help_lbl = pyglet.text.Label("", font_name=FONT, font_size=8, color=C_DIM, multiline=True,
                                          width=width - 12, anchor_y="top", batch=self.batch,
                                          group=pyglet.graphics.Group(order=5))
        self.scroll = 0
        self.content_h = 0
        self.active = None
        self.hover = None
        self.visible = True
        self.win_w = self.win_h = 0

    def set_widgets(self, widgets):
        for w in self.widgets:
            for it in w.items:
                it.delete()
        self.widgets = widgets
        cur = None
        for w in widgets:
            if isinstance(w, Section):
                cur = w
            else:
                w.section = cur
            w.build(self.batch, self.groups)
        self.active = None
        self.layout()

    def layout(self, win_w=None, win_h=None):
        if win_w is not None:
            self.win_w, self.win_h = win_w, win_h
        x0 = self.win_w - self.width
        self.bg.position = (x0, 0)
        self.bg.height = self.win_h
        help_h = 44
        self.help_bg.position = (x0, 0)
        self.help_lbl.x, self.help_lbl.y = x0 + 6, help_h - 4
        top = self.win_h - 6
        y = top + self.scroll
        pad = 8
        total = 0
        for w in self.widgets:
            if not w.visible():
                w.set_visible(False)
                continue
            gap = 6 if isinstance(w, Section) else 1
            y -= w.h + gap
            total += w.h + gap
            w.place(x0 + pad, y, self.width - 2 * pad)
            w.set_visible(help_h <= y and y + w.h <= top)
        self.content_h = total + 10

    def max_scroll(self):
        return max(0, self.content_h - (self.win_h - 50))

    def contains(self, x, y):
        return self.visible and x >= self.win_w - self.width

    def widget_at(self, x, y):
        if y < 44:
            return None
        for w in self.widgets:
            if w.visible() and w.items and w.items[0].visible and w.hit(x, y):
                return w
        return None

    def on_press(self, x, y, button, mods):
        if not self.contains(x, y):
            return False
        w = self.widget_at(x, y)
        if w is not None:
            r = w.press(x, y, button, mods)
            if r == "capture":
                self.active = w
            elif r == "layout":
                self.layout()
        return True

    def on_drag(self, x, y, dx, dy, buttons, mods):
        if self.active is not None:
            self.active.drag(x, y, dx, dy, mods)
            return True
        return self.contains(x, y)

    def on_release(self, x, y, button, mods):
        if self.active is not None:
            self.active.release()
            self.active = None
            return True
        return self.contains(x, y)

    def on_scroll(self, x, y, sy):
        if not self.contains(x, y):
            return False
        self.scroll = min(max(self.scroll - sy * 40, 0), self.max_scroll())
        self.layout()
        return True

    def on_motion(self, x, y):
        w = self.widget_at(x, y) if self.contains(x, y) else None
        if isinstance(w, Row):
            for b in w.buttons:
                if b.hit(x, y):
                    w = b
        if w is not self.hover:
            self.hover = w
            t = (w.help or "") if w is not None else ""
            if self.help_lbl.text != t:
                self.help_lbl.text = t

    def draw(self):
        if not self.visible:
            return
        for w in self.widgets:
            if w.visible() and w.items and w.items[0].visible:
                w.refresh()
        self.batch.draw()


class TimelineBar:
    """bottom bar: |< < > >| buttons, REC, the cache track and a time readout"""
    h = 52

    def __init__(self, app):
        self.app = app
        self.batch = pyglet.graphics.Batch()
        g0, g1, g2, g3 = (pyglet.graphics.Group(order=i) for i in range(4))
        self.bg = shapes.Rectangle(0, 0, 10, self.h, color=(16, 18, 23, 245), batch=self.batch, group=g0)
        self.track = shapes.Rectangle(0, 0, 10, 14, color=C_TRACK, batch=self.batch, group=g1)
        self.pre = shapes.Rectangle(0, 0, 1, 14, color=(70, 56, 90, 255), batch=self.batch, group=g1)
        self.cached = shapes.Rectangle(0, 0, 1, 14, color=(56, 96, 150, 255), batch=self.batch, group=g2)
        self.tend = shapes.Rectangle(0, 0, 2, 20, color=(200, 160, 60, 255), batch=self.batch, group=g3)
        self.head = shapes.Rectangle(0, 0, 3, 24, color=(240, 240, 240, 255), batch=self.batch, group=g3)
        self.live = shapes.Rectangle(0, 0, 2, 18, color=(90, 200, 120, 255), batch=self.batch, group=g3)
        self.lbl = pyglet.text.Label("", font_name=MONO, font_size=9, color=C_TEXT, anchor_y="center",
                                     batch=self.batch, group=g3)
        self.lbl2 = pyglet.text.Label("", font_name=MONO, font_size=8, color=C_DIM, anchor_x="right",
                                      anchor_y="center", batch=self.batch, group=g3)
        self.btns = []
        for name in ("|<", "<", "play", ">", ">|", "rec"):
            b = shapes.Rectangle(0, 0, 30, 26, color=(50, 56, 70, 255), batch=self.batch, group=g1)
            t = pyglet.text.Label(name, font_name=FONT, font_size=10, weight="bold", color=C_TEXT, anchor_x="center",
                                  anchor_y="center", batch=self.batch, group=g3)
            self.btns.append((name, b, t))
        self.ticks = []
        self.tick_group = g2
        self.scrubbing = False
        self.w = 0
        self._last = None

    def layout(self, w):
        self.w = w
        self.bg.width = w
        x = 8
        for name, b, t in self.btns:
            bw = 44 if name in ("play", "rec") else 30
            b.position = (x, 13)
            b.width = bw
            t.x, t.y = x + bw / 2, 26
            x += bw + 4
        self.tx0 = x + 10
        self.tx1 = w - 12
        self.track.position = (self.tx0, 26)
        self.track.width = max(10, self.tx1 - self.tx0)
        self.lbl.x, self.lbl.y = self.tx0, 8
        self.lbl2.x, self.lbl2.y = self.tx1, 8
        self._last = None

    def axis(self):
        """(tau_min, tau_max) of the track"""
        s = self.app.sim
        tl = self.app.timeline
        lo = -s.P["pre_t"] if (s.P["pre_t"] > 0 or (tl.snaps and tl.snaps[0]["tau"] < 0)) else 0.0
        if tl.snaps:
            lo = min(lo, tl.snaps[0]["tau"])
        hi = max(s.P["t_end"], s.tau(), tl.snaps[-1]["tau"] if tl.snaps else 0.0) * 1.02 + 1e-6
        return lo, hi

    def x_of(self, tau):
        lo, hi = self.axis()
        return self.tx0 + (tau - lo) / (hi - lo) * (self.tx1 - self.tx0)

    def tau_of(self, x):
        lo, hi = self.axis()
        return lo + (x - self.tx0) / max(self.tx1 - self.tx0, 1) * (hi - lo)

    def refresh(self):
        app = self.app
        s, tl = app.sim, app.timeline
        lo, hi = self.axis()
        x0 = self.tx0
        y = 26
        self.pre.position = (x0, y)
        self.pre.width = max(0, min(self.x_of(0.0), self.tx1) - x0) if lo < 0 else 0
        if tl.snaps:
            a, b = self.x_of(tl.snaps[0]["tau"]), self.x_of(tl.snaps[-1]["tau"])
            self.cached.position = (a, y)
            self.cached.width = max(2, b - a)
        else:
            self.cached.width = 0
        self.tend.position = (self.x_of(s.P["t_end"]) - 1, y - 3)
        cur = app.view_tau()
        self.head.position = (self.x_of(cur) - 1, y - 5)
        self.live.position = (self.x_of(s.tau()) - 1, y - 2)
        self.live.visible = app.cursor is not None
        for name, b, t in self.btns:
            if name == "play":
                t.text = "||" if app.running else "▶"
            if name == "rec":
                on = app.recorder.active
                b.color = C_REC if on else (50, 56, 70, 255)
                t.text = "● REC" if on else "● rec"
        txt = (f"t={cur:+.4f}" if cur < 0 else f"t={cur:.4f}") + ("  (pre-turbulence)" if cur < 0 else "") + \
              ("  · cache" if app.cursor is not None else "  · live") + \
              (f"   {app.recorder.status()}" if app.recorder.active else "")
        if txt != self.lbl.text:
            self.lbl.text = txt
        t2 = tl.status()
        if t2 != self.lbl2.text:
            self.lbl2.text = t2

    def on_press(self, x, y, button, mods):
        if y > self.h:
            return False
        for name, b, t in self.btns:
            if b.x <= x <= b.x + b.width and b.y <= y <= b.y + b.height:
                self.app.transport(name)
                return True
        if self.tx0 - 6 <= x <= self.tx1 + 6:
            self.scrubbing = True
            self.app.scrub(self.tau_of(x))
        return True

    def on_drag(self, x, y, dx, dy, buttons, mods):
        if self.scrubbing:
            self.app.scrub(self.tau_of(min(max(x, self.tx0), self.tx1)))
            return True
        return y <= self.h

    def on_release(self, x, y, button, mods):
        if self.scrubbing:
            self.scrubbing = False
            return True
        return y <= self.h

    def draw(self):
        self.refresh()
        self.batch.draw()

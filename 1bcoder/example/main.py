#!/usr/bin/env python3
"""Bass Key Piano Tutor — main.py"""
import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.core.window import Window
from kivy.core.text import Label as CoreLabel
from kivy.clock import Clock
from kivy.metrics import sp, dp

Window.size = (400, 700)

QUIZ_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']   # natural notes only

# ── helpers ───────────────────────────────────────────────────────────────────

def draw_text(canvas, text, cx, cy, font_size, color=(0, 0, 0, 1)):
    lbl = CoreLabel(text=text, font_size=font_size)
    lbl.refresh()
    tex = lbl.texture
    with canvas:
        Color(*color)
        Rectangle(
            texture=tex,
            pos=(cx - tex.width / 2, cy - tex.height / 2),
            size=tex.size,
        )


# ── note → staff position (in units of ls above the bottom staff line) ───────
#
#  Bass clef lines (bottom→top): G  B  D  F  A
#  D is on the middle (3rd) line → 2.0 × ls

NOTE_POS = {
    'C':  1.5,
    'C#': 1.5,
    'D':  2.0,
    'D#': 2.0,
    'E':  2.5,
    'F':  3.0,
    'F#': 3.0,
    'G':  3.5,
    'G#': 3.5,
    'A':  4.0,
    'A#': 4.0,
    'B':  4.5,
}


# ── music staff widget ────────────────────────────────────────────────────────

class MusicStaff(Widget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._note  = None
        self._green = False
        self.bind(pos=self._draw, size=self._draw)

    def show_note(self, note, green=False):
        self._note  = note
        self._green = green
        self._draw()

    def _draw(self, *_):
        self.canvas.clear()

        with self.canvas:
            Color(1, 1, 1, 1)
            Rectangle(pos=self.pos, size=self.size)

        ls      = min(self.height * 0.09, dp(22))
        staff_w = self.width * 0.42
        sx      = self.x + (self.width - staff_w) / 2
        sy      = self.y + self.height / 2 - ls * 2   # bottom line y

        # 5 staff lines
        with self.canvas:
            Color(0, 0, 0, 1)
            for i in range(5):
                Line(points=[sx, y := sy + i * ls, sx + staff_w, y], width=dp(1.5))

        # note head
        if self._note and self._note in NOTE_POS:
            r  = ls * 0.44
            cx = sx + staff_w / 2
            cy = sy + NOTE_POS[self._note] * ls
            with self.canvas:
                if self._green:
                    Color(0.15, 0.75, 0.25, 1)
                    Ellipse(pos=(cx - r, cy - r), size=(r * 2, r * 2))
                else:
                    Color(0, 0, 0, 1)
                    Line(circle=(cx, cy, r), width=dp(2))


# ── piano keyboard widget ─────────────────────────────────────────────────────

class PianoKeyboard(Widget):
    WHITE_NOTES   = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    BLACK_INDICES = [0, 1, 3, 4, 5]
    BLACK_NOTES   = ['C#', 'D#', 'F#', 'G#', 'A#']

    def __init__(self, on_note_press=None, **kwargs):
        super().__init__(**kwargs)
        self._callback = on_note_press
        self._pressed  = None
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.clear()
        w  = self.width / 7
        h  = self.height
        bw = w * 0.62
        bh = h * 0.58

        with self.canvas:
            Color(0.55, 0.55, 0.55, 1)
            Rectangle(pos=self.pos, size=self.size)

        for i, note in enumerate(self.WHITE_NOTES):
            x = self.x + i * w
            pressed = self._pressed == note
            with self.canvas:
                Color(0.48, 0.72, 1.0, 1) if pressed else Color(0.97, 0.97, 0.97, 1)
                Rectangle(pos=(x + 2, self.y + 2), size=(w - 4, h - 4))
                Color(0.12, 0.12, 0.12, 1)
                Line(rectangle=(x + 1, self.y + 1, w - 2, h - 2), width=1.2)
            draw_text(
                self.canvas, note,
                x + w / 2, self.y + h * 0.09,
                sp(11),
                (0.1, 0.1, 0.55, 1) if pressed else (0.18, 0.18, 0.18, 1),
            )

        for i, bi in enumerate(self.BLACK_INDICES):
            note = self.BLACK_NOTES[i]
            pressed = self._pressed == note
            bx = self.x + (bi + 1) * w - bw / 2
            by = self.y + h - bh
            with self.canvas:
                Color(0.12, 0.38, 0.78, 1) if pressed else Color(0.07, 0.07, 0.07, 1)
                Rectangle(pos=(bx + 1, by), size=(bw - 2, bh))
            draw_text(
                self.canvas,
                note.replace('#', '♯'),
                bx + bw / 2, by + bh * 0.2,
                sp(9),
                (0.75, 0.88, 1.0, 1) if pressed else (0.80, 0.80, 0.80, 1),
            )

    def _note_at(self, tx, ty):
        rx = tx - self.x
        ry = ty - self.y
        if not (0 <= rx <= self.width and 0 <= ry <= self.height):
            return None
        w  = self.width / 7
        h  = self.height
        bw = w * 0.62
        bh = h * 0.58
        if ry >= h - bh:
            for i, bi in enumerate(self.BLACK_INDICES):
                bx = (bi + 1) * w - bw / 2
                if bx <= rx <= bx + bw:
                    return self.BLACK_NOTES[i]
        idx = int(rx / w)
        return self.WHITE_NOTES[idx] if 0 <= idx < 7 else None

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        note = self._note_at(touch.x, touch.y)
        if note:
            self._pressed = note
            self._redraw()
            if self._callback:
                self._callback(note)
            Clock.schedule_once(lambda _: self._release(), 0.18)
        return True

    def _release(self):
        self._pressed = None
        self._redraw()


# ── app ───────────────────────────────────────────────────────────────────────

class PianoTutorApp(App):
    title = 'Bass Key Piano Tutor'

    def build(self):
        Window.clearcolor = (1, 1, 1, 1)

        self._score        = 0
        self._current_note = None
        self._waiting      = False   # True while green note is shown

        root = BoxLayout(orientation='vertical')

        # score label
        self.score_label = Label(
            text='Score: 0',
            size_hint=(1, 0.07),
            font_size=sp(18),
            color=(0, 0, 0, 1),
            bold=True,
        )
        with self.score_label.canvas.before:
            Color(0.93, 0.93, 0.93, 1)
            self._score_bg = Rectangle(
                pos=self.score_label.pos, size=self.score_label.size
            )
        self.score_label.bind(
            pos =lambda w, v: setattr(self._score_bg, 'pos',  v),
            size=lambda w, v: setattr(self._score_bg, 'size', v),
        )
        root.add_widget(self.score_label)

        # staff
        self.staff = MusicStaff(size_hint=(1, 0.58))
        root.add_widget(self.staff)

        # piano
        root.add_widget(PianoKeyboard(
            on_note_press=self._on_note,
            size_hint=(1, 0.35),
        ))

        Clock.schedule_once(lambda _: self._next_note(), 0.3)
        return root

    def _next_note(self):
        self._waiting      = False
        self._current_note = random.choice(QUIZ_NOTES)
        self.staff.show_note(self._current_note, green=False)

    def _on_note(self, pressed):
        if self._waiting:
            return
        if pressed == self._current_note:
            self._score += 1
            self.score_label.text = f'Score: {self._score}'
            self.staff.show_note(self._current_note, green=True)
            self._waiting = True
            Clock.schedule_once(lambda _: self._next_note(), 0.7)


if __name__ == '__main__':
    PianoTutorApp().run()

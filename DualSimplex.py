import numpy as np
import tkinter as tk
from tkinter import messagebox
import random
import math

# matplotlib embedded in tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

BG       = "#0f1117"
PANEL    = "#1a1d2e"
PANEL2   = "#232640"
BORDER   = "#2e3357"
ACCENT   = "#5e81f4"
ACCENT2  = "#f4a25e"
SUCCESS  = "#4ecdc4"
TEXT     = "#e8eaf6"
SUBTEXT  = "#8b90b8"
ENTRY_BG = "#12152a"
RED      = "#f45e7a"
WARN     = "#f4d03f"

HL_ROW_BG  = "#2a2000";  HL_ROW_FG  = "#f4d03f"
HL_COL_BG  = "#001a2a";  HL_COL_FG  = "#5eb8f4"
HL_CELL_BG = "#3a1500";  HL_CELL_FG = "#ff8c42"

MPL_BG     = "#0f1117"
MPL_AX     = "#1a1d2e"
MPL_TEXT   = "#e8eaf6"
MPL_GRID   = "#2e3357"

SIDEBAR_W  = 200   # px


def _lighten(hex_color, amt=30):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(min(255,r+amt), min(255,g+amt), min(255,b+amt))


def styled_btn(parent, text, cmd, color, width=20, pady=7, padx=14):
    btn = tk.Button(parent, text=text, command=cmd,
                    bg=color, fg="#ffffff",
                    font=("Consolas", 10, "bold"),
                    relief="flat", bd=0, cursor="hand2",
                    activebackground=color, activeforeground="#ffffff",
                    padx=padx, pady=pady, width=width)
    btn.bind("<Enter>", lambda e: btn.config(bg=_lighten(color)))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn


def make_entry(parent, width=6):
    return tk.Entry(parent, width=width,
                    bg=ENTRY_BG, fg="#ffffff",
                    insertbackground=ACCENT,
                    font=("Consolas", 11),
                    relief="flat", bd=0,
                    highlightthickness=1,
                    highlightbackground=BORDER,
                    highlightcolor=ACCENT,
                    justify="center")


def _fmt_row(row, b, sense, n):
    terms = " + ".join(f"{row[j]:.4g}·x{j+1}" for j in range(n))
    return f"{terms}  {sense}  {b:.4g}"
    
class DualSimplexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Simplex Method Solver")
        self.root.geometry("1600x920")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.n = self.m = 0
        self.c_entries  = []
        self.A_entries  = []
        self.b_entries  = []
        self.sense_vars = []
        self.obj_var    = tk.StringVar(value="Minimize")

        # state kept after solve for visualizations
        self._sol_state  = None   # dict filled after a successful solve
        self._active_viz = tk.StringVar(value="")
        self._last_ex_idx = -1   # track last example so we don't repeat

        self._build_ui()

    # ═══════════════════════════════════════════════════════════════════════════
    #  UI LAYOUT
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # title bar
        bar = tk.Frame(self.root, bg=PANEL, pady=11)
        bar.pack(fill="x")
        tk.Label(bar, text="⬡  DUAL SIMPLEX METHOD SOLVER",
                 bg=PANEL, fg=ACCENT, font=("Consolas", 16, "bold")).pack(side="left", padx=22)
        tk.Label(bar, text="Operations Research  •  Mini Project",
                 bg=PANEL, fg=SUBTEXT, font=("Consolas", 10)).pack(side="right", padx=22)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=10)

        # ── Left input panel ──
        self.left_panel = tk.Frame(body, bg=PANEL,
                                   highlightthickness=1, highlightbackground=BORDER)
        self.left_panel.pack(side="left", fill="y", ipadx=10, ipady=10, padx=(0, 8))

        # ── Centre solution panel ──
        centre = tk.Frame(body, bg=PANEL,
                          highlightthickness=1, highlightbackground=BORDER)
        centre.pack(side="left", fill="both", expand=True, ipadx=10, ipady=10, padx=(0, 8))

        # ── Right visualisation sidebar (hidden until after first solve) ──
        self.viz_panel = tk.Frame(body, bg=PANEL,
                                  highlightthickness=1, highlightbackground=BORDER,
                                  width=SIDEBAR_W)
        self.viz_panel.pack_propagate(False)
        # NOTE: intentionally NOT packed here — revealed by _reveal_sidebar()

        self._body = body   # keep ref so we can pack into it later

        self._build_left(self.left_panel)
        self._build_centre(centre)
        self._build_viz_sidebar(self.viz_panel)

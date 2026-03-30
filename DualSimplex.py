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

        # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left(self, p):
        self._section(p, "STEP 1 — Problem Size")
        size = tk.Frame(p, bg=PANEL2, highlightthickness=1, highlightbackground=BORDER)
        size.pack(fill="x", padx=12, pady=4, ipady=8)

        r1 = tk.Frame(size, bg=PANEL2); r1.pack(fill="x", padx=12, pady=3)
        tk.Label(r1, text="Variables  (x₁, x₂, …):",
                 bg=PANEL2, fg=TEXT, font=("Consolas", 10)).pack(side="left")
        self.var_entry = make_entry(r1, width=4); self.var_entry.pack(side="right")

        r2 = tk.Frame(size, bg=PANEL2); r2.pack(fill="x", padx=12, pady=3)
        tk.Label(r2, text="Constraints:",
                 bg=PANEL2, fg=TEXT, font=("Consolas", 10)).pack(side="left")
        self.con_entry = make_entry(r2, width=4); self.con_entry.pack(side="right")

        self._section(p, "Objective Direction")
        obj_frame = tk.Frame(p, bg=PANEL2, highlightthickness=1, highlightbackground=BORDER)
        obj_frame.pack(fill="x", padx=12, pady=4, ipady=6)
        for opt, color in [("Minimize", SUCCESS), ("Maximize", ACCENT2)]:
            tk.Radiobutton(obj_frame, text=opt, variable=self.obj_var, value=opt,
                           bg=PANEL2, fg=color, selectcolor=PANEL2,
                           activebackground=PANEL2, activeforeground=color,
                           font=("Consolas", 10, "bold"),
                           indicatoron=1, relief="flat", cursor="hand2"
                           ).pack(side="left", padx=18, pady=4)

        styled_btn(p, "▶  Generate Input Fields",
                   self.create_inputs, ACCENT, width=23).pack(pady=(10,4), padx=12)

        # scrollable input area
        cw = tk.Frame(p, bg=PANEL); cw.pack(fill="both", expand=True, padx=12, pady=4)
        self.canvas = tk.Canvas(cw, bg=PANEL, highlightthickness=0, width=420)
        vsb = tk.Scrollbar(cw, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inp_frame = tk.Frame(self.canvas, bg=PANEL)
        self.canvas.create_window((0,0), window=self.inp_frame, anchor="nw")
        self.inp_frame.bind("<Configure>",
                            lambda e: self.canvas.configure(
                                scrollregion=self.canvas.bbox("all")))

        styled_btn(p, "⚡  Solve — Show All Steps",
                   self.solve_steps, SUCCESS, width=23).pack(pady=(6,14), padx=12)
    # ── Centre panel ──────────────────────────────────────────────────────────

    def _build_centre(self, p):
        header = tk.Frame(p, bg=PANEL); header.pack(fill="x", padx=12, pady=(12,4))
        tk.Label(header, text="SOLUTION  &  STEPS",
                 bg=PANEL, fg=ACCENT, font=("Consolas", 11, "bold")).pack(side="left")
        styled_btn(header, "🗑  Clear", self._clear,
                   "#2e3357", width=10, pady=4).pack(side="right")

        legend = tk.Frame(p, bg=PANEL); legend.pack(fill="x", padx=12, pady=(0,4))
        def swatch(bg, fg, label):
            f = tk.Frame(legend, bg=PANEL); f.pack(side="left", padx=(0,16))
            tk.Label(f, text="  ", bg=bg, width=2).pack(side="left")
            tk.Label(f, text=f" {label}", bg=PANEL, fg=fg,
                     font=("Consolas", 9)).pack(side="left")
        swatch(HL_ROW_BG, HL_ROW_FG,  "Pivot row")
        swatch(HL_COL_BG, HL_COL_FG,  "Pivot column")
        swatch(HL_CELL_BG,HL_CELL_FG, "Pivot element")

        wrap = tk.Frame(p, bg=ENTRY_BG,
                        highlightthickness=1, highlightbackground=BORDER)
        wrap.pack(fill="both", expand=True, padx=12, pady=6)

        self.out = tk.Text(wrap, bg=ENTRY_BG, fg=TEXT,
                           font=("Consolas", 10), relief="flat", bd=0,
                           padx=10, pady=8, wrap="none", state="disabled")
        vsb2 = tk.Scrollbar(wrap, orient="vertical",   command=self.out.yview)
        hsb2 = tk.Scrollbar(wrap, orient="horizontal", command=self.out.xview)
        self.out.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)
        hsb2.pack(side="bottom", fill="x")
        vsb2.pack(side="right",  fill="y")
        self.out.pack(fill="both", expand=True)

        for tag, fg, font in [
            ("head",    ACCENT,   ("Consolas",11,"bold")),
            ("iter",    ACCENT2,  ("Consolas",10,"bold")),
            ("pivot",   ACCENT2,  ("Consolas",10)),
            ("rowop",   SUBTEXT,  ("Consolas",10)),
            ("tbl",     "#c5cae9",("Consolas",10)),
            ("sol",     SUCCESS,  ("Consolas",11,"bold")),
            ("zval",    ACCENT2,  ("Consolas",12,"bold")),
            ("err",     RED,      ("Consolas",10,"bold")),
            ("dim",     SUBTEXT,  ("Consolas",10)),
            ("convert", WARN,     ("Consolas",10,"bold")),
            ("convdim", WARN,     ("Consolas",10)),
        ]:
            self.out.tag_configure(tag, foreground=fg, font=font)

        self.out.tag_configure("hl_row",  background=HL_ROW_BG,  foreground=HL_ROW_FG,  font=("Consolas",10,"bold"))
        self.out.tag_configure("hl_col",  background=HL_COL_BG,  foreground=HL_COL_FG,  font=("Consolas",10))
        self.out.tag_configure("hl_cell", background=HL_CELL_BG, foreground=HL_CELL_FG, font=("Consolas",10,"bold"))
        self.out.tag_configure("hl_zcol", background=HL_COL_BG,  foreground=HL_COL_FG,  font=("Consolas",10))

# ── Viz sidebar ───────────────────────────────────────────────────────────

    def _build_viz_sidebar(self, p):
        tk.Label(p, text="VISUALIZATIONS", bg=PANEL, fg=ACCENT,
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=10, pady=(14,8))

        self._viz_buttons = {}
        viz_items = [
            ("feasible",   "📐  Feasible Region"),
            ("path",       "🔀  Simplex Path"),
            ("heatmap",    "🌡  Tableau Heatmap"),
            ("inspector",  "🔍  Constraint Info"),
            ("objslider",  "🎚  Objective Slider"),
        ]

        for key, label in viz_items:
            btn = tk.Button(p, text=label, bg=PANEL2, fg=SUBTEXT,
                            font=("Consolas", 9), relief="flat", bd=0,
                            cursor="hand2", anchor="w", padx=10, pady=8,
                            width=22,
                            command=lambda k=key: self._show_viz(k))
            btn.pack(fill="x", padx=8, pady=2)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=BORDER))
            btn.bind("<Leave>", lambda e, b=btn, k=key: b.config(
                bg=ACCENT if self._active_viz.get()==k else PANEL2))
            self._viz_buttons[key] = btn

        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=8, pady=10)

        # placeholder label
        self._viz_hint = tk.Label(p, text="Solve a problem\nto unlock\nvisualizations",
                                  bg=PANEL, fg=SUBTEXT,
                                  font=("Consolas", 9), justify="center")
        self._viz_hint.pack(pady=20)

        # disable all buttons initially
        self._set_viz_buttons_state("disabled")

    def _set_viz_buttons_state(self, state):
        for btn in self._viz_buttons.values():
            btn.config(state=state)

    def _set_active_btn(self, key):
        self._active_viz.set(key)
        for k, btn in self._viz_buttons.items():
            btn.config(bg=ACCENT if k == key else PANEL2,
                       fg=TEXT   if k == key else SUBTEXT)

    def _reveal_sidebar(self):
        """Pack the sidebar the first time a solve succeeds."""
        if not self.viz_panel.winfo_ismapped():
            self.viz_panel.pack(side="left", fill="y", ipadx=6, ipady=10)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=ACCENT,
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=14, pady=(12,4))

    # ═══════════════════════════════════════════════════════════════════════════
    #  INPUT GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def create_inputs(self):
        for w in self.inp_frame.winfo_children():
            w.destroy()
        self.c_entries  = []
        self.A_entries  = []
        self.b_entries  = []
        self.sense_vars = []

        try:
            self.n = int(self.var_entry.get())
            self.m = int(self.con_entry.get())
            assert 1 <= self.n <= 10 and 1 <= self.m <= 10
        except Exception:
            messagebox.showerror("Error", "Enter integers 1–10 for both fields.")
            return

        n, m = self.n, self.m
        f    = self.inp_frame
        obj_dir = self.obj_var.get()

        tk.Label(f, text=f"STEP 2 — Objective  ({obj_dir}  Z = c·x)",
                 bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold")).grid(
            row=0, column=0, columnspan=2*n+4, sticky="w", padx=4, pady=(10,2))

        tk.Label(f, text=f"{obj_dir[:3].upper()} Z =",
                 bg=PANEL, fg=SUBTEXT, font=("Consolas", 10)).grid(row=1, column=0, padx=4)

        for j in range(n):
            col  = 1 + j*2
            cell = tk.Frame(f, bg=PANEL); cell.grid(row=1, column=col, padx=2, pady=3)
            e = make_entry(cell, width=5); e.pack()
            tk.Label(cell, text=f"x{j+1}", bg=PANEL, fg=SUBTEXT,
                     font=("Consolas", 8)).pack()
            self.c_entries.append(e)
            if j < n-1:
                tk.Label(f, text="+", bg=PANEL, fg=SUBTEXT,
                         font=("Consolas", 11)).grid(row=1, column=col+1)

        tk.Label(f, text="STEP 3 — Constraints  (choose type per row)",
                 bg=PANEL, fg=ACCENT, font=("Consolas", 9, "bold")).grid(
            row=3, column=0, columnspan=2*n+4, sticky="w", padx=4, pady=(12,2))

        tk.Label(f, text="", bg=PANEL).grid(row=4, column=0)
        for j in range(n):
            tk.Label(f, text=f"x{j+1}", bg=PANEL, fg=SUBTEXT,
                     font=("Consolas", 9)).grid(row=4, column=1+j*2)
        tk.Label(f, text=" Type", bg=PANEL, fg=ACCENT2,
                 font=("Consolas", 9, "bold")).grid(row=4, column=2*n+1)
        tk.Label(f, text="  b", bg=PANEL, fg=ACCENT2,
                 font=("Consolas", 9, "bold")).grid(row=4, column=2*n+2)

        for i in range(m):
            bg_r = PANEL if i%2==0 else PANEL2
            tk.Label(f, text=f"R{i+1}", bg=bg_r, fg=SUBTEXT,
                     font=("Consolas", 10)).grid(row=5+i, column=0, padx=5, pady=3)
            row_e = []
            for j in range(n):
                col = 1+j*2
                e = make_entry(f, width=5)
                e.grid(row=5+i, column=col, padx=2, pady=3)
                row_e.append(e)
                if j < n-1:
                    tk.Label(f, text="+", bg=bg_r, fg=SUBTEXT,
                             font=("Consolas",10)).grid(row=5+i, column=col+1)
            self.A_entries.append(row_e)

            sv = tk.StringVar(value="≥")
            self.sense_vars.append(sv)
            om = tk.OptionMenu(f, sv, "≤","≥","=")
            om.config(bg=PANEL2, fg=ACCENT2, activebackground=BORDER,
                      activeforeground=ACCENT2, font=("Consolas",10,"bold"),
                      relief="flat", bd=0, highlightthickness=1,
                      highlightbackground=BORDER, width=3, cursor="hand2")
            om["menu"].config(bg=PANEL2, fg=ACCENT2,
                              activebackground=BORDER, activeforeground=ACCENT2,
                              font=("Consolas",10))
            om.grid(row=5+i, column=2*n+1, padx=6, pady=3)

            b_e = make_entry(f, width=5)
            b_e.grid(row=5+i, column=2*n+2, padx=3, pady=3)
            self.b_entries.append(b_e)

        styled_btn(f, "📋  Random Example", self._load_example,
                   "#2e3357", width=18, pady=4).grid(
            row=6+m, column=0, columnspan=2*n+4, pady=(12,4))

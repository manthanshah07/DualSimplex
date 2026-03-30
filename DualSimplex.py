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

# ── Predefined rich examples (n, m, obj, c, A, b, senses) ─────────────────────
EXAMPLES = [
    # 2-var examples (good for graph view)
    dict(n=2, m=3, obj="Minimize",
         c=[3, 2],
         A=[[1,1],[2,1],[1,3]],
         b=[4, 6, 6],
         s=["≥","≥","≥"]),
    dict(n=2, m=3, obj="Maximize",
         c=[5, 4],
         A=[[6,4],[1,2],[1,0]],
         b=[24,6,4],
         s=["≤","≤","≤"]),
    dict(n=2, m=3, obj="Minimize",
         c=[2, 5],
         A=[[1,0],[0,1],[1,1]],
         b=[4, 6, 7],
         s=["≥","≥","≥"]),
    dict(n=2, m=4, obj="Maximize",
         c=[3, 5],
         A=[[1,0],[0,2],[3,2],[1,1]],
         b=[4, 12, 18, 6],
         s=["≤","≤","≤","≤"]),
    dict(n=2, m=3, obj="Minimize",
         c=[4, 3],
         A=[[2,1],[1,2],[1,1]],
         b=[8, 8, 5],
         s=["≥","≥","="]),
    # 3-var examples
    dict(n=3, m=3, obj="Minimize",
         c=[2, 3, 1],
         A=[[1,2,1],[2,1,0],[0,1,2]],
         b=[6, 8, 4],
         s=["≥","≥","≥"]),
    dict(n=3, m=4, obj="Maximize",
         c=[5, 4, 3],
         A=[[6,4,2],[3,2,5],[5,6,5],[8,2,4]],
         b=[240,270,420,350],
         s=["≤","≤","≤","≤"]),
    dict(n=3, m=3, obj="Minimize",
         c=[1, 2, 3],
         A=[[1,1,0],[0,1,1],[1,0,1]],
         b=[3, 4, 5],
         s=["≥","≥","≥"]),
    dict(n=3, m=3, obj="Maximize",
         c=[7, 5, 3],
         A=[[1,1,1],[2,1,0],[0,1,2]],
         b=[10,14,14],
         s=["≤","≤","≤"]),
    dict(n=3, m=4, obj="Minimize",
         c=[3, 2, 4],
         A=[[1,1,1],[2,0,1],[0,2,1],[1,2,0]],
         b=[5, 6, 4, 4],
         s=["≥","≥","≥","≥"]),
    # Mixed constraint examples
    dict(n=2, m=3, obj="Maximize",
         c=[6, 5],
         A=[[1,1],[3,2],[1,3]],
         b=[8, 18, 12],
         s=["≤","≤","≤"]),
    dict(n=3, m=3, obj="Minimize",
         c=[2, 1, 3],
         A=[[1,2,1],[2,1,2],[1,1,1]],
         b=[8, 10, 5],
         s=["≥","≥","="]),
]

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


    # ── Random example loader ─────────────────────────────────────────────────

    def _load_example(self):
        if not self.c_entries:
            messagebox.showinfo("Example", "Generate input fields first.")
            return

        # Filter examples that fit current n, m
        fitting = [e for e in EXAMPLES if e["n"] == self.n and e["m"] == self.m]
        if not fitting:
            # Generate a random valid example for any n, m
            ex = self._generate_random_example(self.n, self.m)
        else:
            # Pick randomly but avoid repeating last
            choices = [i for i,e in enumerate(fitting) if i != self._last_ex_idx]
            if not choices:
                choices = list(range(len(fitting)))
            idx = random.choice(choices)
            self._last_ex_idx = idx
            ex = fitting[idx]

        # Apply
        self.obj_var.set(ex["obj"])
        for j, e in enumerate(self.c_entries):
            e.delete(0, tk.END); e.insert(0, ex["c"][j])
        for i in range(self.m):
            for j, e in enumerate(self.A_entries[i]):
                e.delete(0, tk.END); e.insert(0, ex["A"][i][j])
            self.b_entries[i].delete(0, tk.END)
            self.b_entries[i].insert(0, ex["b"][i])
            self.sense_vars[i].set(ex["s"][i])

    def _generate_random_example(self, n, m):
        """Procedurally generate a well-formed random LP example."""
        obj = random.choice(["Minimize", "Maximize"])
        c   = [random.randint(1, 8) for _ in range(n)]
        senses = [random.choice(["≤","≥"]) for _ in range(m)]
        A, b = [], []
        for i in range(m):
            row = [random.randint(1, 6) for _ in range(n)]
            # make RHS sensible: sum of row * small factor
            rhs = max(1, int(sum(row) * random.uniform(0.8, 2.5)))
            A.append(row)
            b.append(rhs)
        return dict(n=n, m=m, obj=obj, c=c, A=A, b=b, s=senses)


    # ═══════════════════════════════════════════════════════════════════════════
    #  OUTPUT HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _clear(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", tk.END)
        self.out.configure(state="disabled")

    def put(self, text, tag=""):
        self.out.configure(state="normal")
        self.out.insert(tk.END, text+"\n", tag if tag else ())
        self.out.configure(state="disabled")
        self.out.see(tk.END)

    def fmt_table(self, tableau, basis, n, z_row,
                  pivot_row=-1, pivot_col=-1, ratios=None):
        self.out.configure(state="normal")
        total = tableau.shape[1] - 1

        hdr = f"{'Basic':<9}│"
        for j in range(total):
            lbl = f"x{j+1}" if j < n else f"s{j-n+1}"
            hdr += f"{lbl:>8}"
        hdr += "  │" + f"{'RHS':>9}"
        div = "─" * len(hdr)

        self.out.insert(tk.END, hdr+"\n", "tbl")
        self.out.insert(tk.END, div+"\n", "tbl")

        self.out.insert(tk.END, f"{'Z':<9}│", "tbl")
        for j in range(total):
            tag = "hl_zcol" if j == pivot_col else "tbl"
            self.out.insert(tk.END, f"{z_row[j]:8.3f}", tag)
        self.out.insert(tk.END, "  │"+f"{z_row[-1]:9.3f}\n", "tbl")
        self.out.insert(tk.END, div+"\n", "tbl")

        for i, row in enumerate(tableau):
            bv  = basis[i]
            lbl = f"x{bv+1}" if bv < n else f"s{bv-n+1}"
            ipr = (i == pivot_row)
            rt  = "hl_row" if ipr else "tbl"
            self.out.insert(tk.END, f"{lbl:<9}│", rt)
            for j in range(total):
                if ipr and j == pivot_col: tag = "hl_cell"
                elif ipr:                  tag = "hl_row"
                elif j == pivot_col:       tag = "hl_col"
                else:                      tag = "tbl"
                self.out.insert(tk.END, f"{row[j]:8.3f}", tag)
            self.out.insert(tk.END, "  │"+f"{row[-1]:9.3f}\n", rt)

        self.out.insert(tk.END, div+"\n", "tbl")

        if ratios is not None:
            self.out.insert(tk.END, f"{'Ratio':<9}│", "tbl")
            for j in range(len(ratios)):
                txt = "    ∞   " if ratios[j]==np.inf else f"{ratios[j]:8.3f}"
                self.out.insert(tk.END, txt, "hl_col" if j==pivot_col else "tbl")
            self.out.insert(tk.END, "\n")
            self.out.insert(tk.END, div+"\n", "tbl")

        self.out.configure(state="disabled")
        self.out.see(tk.END)

    # ═══════════════════════════════════════════════════════════════════════════
    #  CONVERSION
    # ═══════════════════════════════════════════════════════════════════════════

    def _convert_to_standard(self, c_orig, A_orig, b_orig, senses, obj_dir):
        log = []
        n   = len(c_orig)
        c   = c_orig.copy()
        A   = [row.copy() for row in A_orig]
        b   = b_orig.copy()
        sns = list(senses)

        log.append(("─"*68, "convdim"))
        log.append(("  STANDARD FORM CONVERSION", "convert"))
        log.append(("─"*68, "convdim"))

        if obj_dir == "Maximize":
            log.append(("", ""))
            log.append(("  Objective: MAXIMIZE → convert to MINIMIZE", "convert"))
            log.append(("  Strategy : Multiply objective coefficients by −1", "convdim"))
            log.append(("           : Max Z = c·x  ≡  Min Z' = −c·x", "convdim"))
            log.append(("  Original c = [" + ", ".join(f"{v:.4g}" for v in c) + "]", "convdim"))
            c = -c
            log.append(("  Negated  c = [" + ", ".join(f"{v:.4g}" for v in c) + "]", "convdim"))
            log.append(("  ✔ Objective converted.", "convert"))
        else:
            log.append(("", ""))
            log.append(("  Objective: MINIMIZE — no conversion needed.", "convdim"))

        log.append(("", ""))
        log.append(("  Constraints: converting each to  ≥  form", "convert"))
        log.append(("─"*40, "convdim"))

        new_A, new_b, new_sns = [], [], []
        for i, (row, bi, sense) in enumerate(zip(A, b, sns)):
            label = f"  R{i+1}:"
            if sense == "≥":
                log.append((f"{label}  already  ≥  — no change.", "convdim"))
                new_A.append(row); new_b.append(bi); new_sns.append("≥")
            elif sense == "≤":
                log.append((f"{label}  ≤  → multiply both sides by −1", "convdim"))
                log.append((f"       Original: {_fmt_row(row, bi, '≤', n)}", "convdim"))
                fr = [-v for v in row]; fb = -bi
                log.append((f"       Flipped : {_fmt_row(fr, fb, '≥', n)}", "convdim"))
                new_A.append(fr); new_b.append(fb); new_sns.append("≥")
            elif sense == "=":
                log.append((f"{label}  =  → split into two  ≥  constraints", "convdim"))
                log.append((f"       Original: {_fmt_row(row, bi, '=', n)}", "convdim"))
                log.append((f"       Split 1 : {_fmt_row(row, bi, '≥', n)}", "convdim"))
                fr = [-v for v in row]; fb = -bi
                log.append((f"       Split 2 : {_fmt_row(fr, fb, '≥', n)}", "convdim"))
                new_A.append(row);  new_b.append(bi); new_sns.append("≥")
                new_A.append(fr);   new_b.append(fb); new_sns.append("≥")

        if any(s in ("≤","=") for s in sns):
            log.append(("", ""))
            log.append(("  ✔ All constraints now in  ≥  form.", "convert"))
        else:
            log.append(("  ✔ All constraints already in  ≥  form.", "convert"))

        log.append(("─"*68, "convdim"))
        log.append(("", ""))

        return c, np.array(new_A, dtype=float), np.array(new_b, dtype=float), log

    # ═══════════════════════════════════════════════════════════════════════════
    #  SOLVER
    # ═══════════════════════════════════════════════════════════════════════════

    def solve_steps(self):
        self._clear()
        self._sol_state = None
        self._set_viz_buttons_state("disabled")

        try:
            c_orig = np.array([float(e.get()) for e in self.c_entries])
            A_orig = [[float(e.get()) for e in row] for row in self.A_entries]
            b_orig = np.array([float(e.get()) for e in self.b_entries])
            senses = [sv.get() for sv in self.sense_vars]
        except ValueError:
            self.put("⚠  Fill ALL fields with numbers before solving.", "err")
            return

        obj_dir = self.obj_var.get()
        n_orig  = self.n

        self.put("━"*68, "head")
        self.put("  DUAL SIMPLEX METHOD — STEP BY STEP SOLUTION", "head")
        self.put("━"*68, "head")
        self.put(f"\n  ORIGINAL PROBLEM  ({obj_dir} Z)", "head")

        obj_str = "  " + obj_dir + " Z = " + " + ".join(
            f"{c_orig[j]:.4g}·x{j+1}" for j in range(n_orig))
        self.put(obj_str, "dim")
        self.put("  Subject to:", "dim")
        for i in range(self.m):
            self.put("    " + f"R{i+1}: " +
                     _fmt_row(A_orig[i], b_orig[i], senses[i], n_orig), "dim")
        self.put("", "")

        c, A, b, conv_log = self._convert_to_standard(
            c_orig, A_orig, b_orig, senses, obj_dir)
        for msg, tag in conv_log:
            self.put(msg, tag)

        m, n = A.shape
        self.put(f"  Variables: {n}    Constraints (after conversion): {m}", "dim")
        self.put("  Minimise Z' = " +
                 " + ".join(f"{c[j]:.4g}·x{j+1}" for j in range(n)), "dim")

        tableau = np.hstack((-A, np.eye(m), (-b).reshape(-1,1)))
        z_row   = np.hstack((-c, np.zeros(m), 0.0))
        basis   = list(range(n, n+m))

        self.put("\n  Initial Tableau (after conversion):\n", "head")
        self.fmt_table(tableau, basis, n, z_row)

        # save snapshots for heatmap / path
        tableaux_history = [tableau.copy()]
        z_row_history    = [z_row.copy()]
        basis_history    = [basis.copy()]

        itr = 0
        while np.any(tableau[:, -1] < -1e-9):
            itr += 1
            if itr > 50:
                self.put("\n⚠  Max iterations reached.", "err"); return

            self.put(f"\n{'━'*68}", "iter")
            self.put(f"  ITERATION {itr}", "iter")
            self.put(f"{'━'*68}", "iter")

            pr = int(np.argmin(tableau[:, -1]))
            self.put(f"\n  ▸ Most negative RHS → Pivot Row = R{pr+1}"
                     f"  (value = {tableau[pr,-1]:.3f})", "pivot")

            ratios = []
            for j in range(tableau.shape[1]-1):
                if tableau[pr, j] < -1e-9:
                    ratio = z_row[j] / tableau[pr, j]
                    ratios.append(ratio if ratio > 1e-9 else np.inf)
                else:
                    ratios.append(np.inf)

            if all(r == np.inf for r in ratios):
                self.put("\n  ✗  No valid pivot — Problem is INFEASIBLE.", "err"); return

            pc    = int(np.argmin(ratios))
            pivot = tableau[pr, pc]
            clbl  = f"x{pc+1}" if pc < n else f"s{pc-n+1}"

            self.put(f"  ▸ Min-ratio test  →  Pivot Col = {clbl}  (col {pc+1})", "pivot")
            self.put(f"  ▸ Pivot Element   =  {pivot:.4f}", "pivot")
            self.put("\n  Tableau with pivot highlighted:\n", "dim")
            self.fmt_table(tableau, basis, n, z_row,
                           pivot_row=pr, pivot_col=pc, ratios=ratios)

            self.put("\n  Row Operations:", "dim")
            self.put(f"    R{pr+1}  ←  R{pr+1} ÷ ({pivot:.4f})", "rowop")
            tableau[pr] /= pivot
            z_row -= z_row[pc] * tableau[pr]

            for i in range(m):
                if i != pr:
                    f_ = tableau[i, pc]
                    if abs(f_) > 1e-12:
                        self.put(f"    R{i+1}  ←  R{i+1} − ({f_:.4f}) × R{pr+1}", "rowop")
                        tableau[i] -= f_ * tableau[pr]

            basis[pr] = pc
            blbl = ", ".join(f"x{b+1}" if b < n else f"s{b-n+1}" for b in basis)
            self.put(f"\n  Updated Basis: [{blbl}]", "dim")
            self.put("\n  Tableau after pivot:\n", "dim")
            self.fmt_table(tableau, basis, n, z_row)

            tableaux_history.append(tableau.copy())
            z_row_history.append(z_row.copy())
            basis_history.append(basis.copy())

        # ── Optimal ──
        self.put(f"\n{'━'*68}", "sol")
        self.put("  ✔  OPTIMAL SOLUTION FOUND", "sol")
        self.put(f"{'━'*68}\n", "sol")

        sol = np.zeros(n + m)
        for idx, bv in enumerate(basis):
            sol[bv] = tableau[idx, -1]
        for i in range(n):
            self.put(f"    x{i+1}  =  {sol[i]:.4f}", "sol")

        z_prime = np.dot(c, sol[:n])
        z_orig  = -z_prime if obj_dir == "Maximize" else z_prime
        obj_lbl = "Z (max)" if obj_dir == "Maximize" else "Z (min)"

        if obj_dir == "Maximize":
            self.put(f"\n    Z' (min, converted)  =  {z_prime:.4f}", "dim")
            self.put(f"    {obj_lbl}  =  −Z'  =  {z_orig:.4f}", "zval")
        else:
            self.put(f"\n    {obj_lbl}  =  {z_orig:.4f}", "zval")
        self.put("")

        # ── Store state for visualizations ──
        self._sol_state = dict(
            n=n, m=m, n_orig=n_orig,
            c=c, c_orig=c_orig,
            A=A, A_orig=np.array(A_orig, dtype=float),
            b=b, b_orig=b_orig,
            senses=senses,
            obj_dir=obj_dir,
            sol=sol,
            z_orig=z_orig,
            basis=basis,
            tableau=tableau,
            z_row=z_row,
            tableaux_history=tableaux_history,
            z_row_history=z_row_history,
            basis_history=basis_history,
        )

        # reveal & unlock sidebar
        self._reveal_sidebar()
        self._set_viz_buttons_state("normal")
        # reset any previously active button highlight for the new solve
        self._active_viz.set("")
        for btn in self._viz_buttons.values():
            btn.config(bg=PANEL2, fg=SUBTEXT)
        self._viz_hint.config(text="Click a view\nto explore", fg=SUCCESS)

    
    # ═══════════════════════════════════════════════════════════════════════════
    #  VISUALIZATION DISPATCHER
    # ═══════════════════════════════════════════════════════════════════════════

    def _show_viz(self, key):
        if self._sol_state is None:
            return
        self._set_active_btn(key)

        s   = self._sol_state
        n   = s["n_orig"]
        is2 = (n == 2)

        graph_views = {"feasible", "path", "objslider"}
        if key in graph_views and not is2:
            messagebox.showinfo("Graph View",
                "Graph visualizations are only available for 2-variable problems.\n"
                "Your problem has more than 2 variables.")
            return

        win = tk.Toplevel(self.root)
        win.configure(bg=MPL_BG)
        win.resizable(True, True)

        # When the window closes, un-highlight the button
        def _on_close():
            if self._active_viz.get() == key:
                self._active_viz.set("")
                self._viz_buttons[key].config(bg=PANEL2, fg=SUBTEXT)
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

        if key == "feasible":
            win.title("Feasible Region")
            win.geometry("720x600")
            self._viz_feasible(win, s)
        elif key == "path":
            win.title("Simplex Path")
            win.geometry("720x600")
            self._viz_path(win, s)
        elif key == "heatmap":
            win.title("Tableau Heatmap")
            win.geometry("820x560")
            self._viz_heatmap(win, s)
        elif key == "inspector":
            win.title("Constraint Inspector")
            win.geometry("560x480")
            self._viz_inspector(win, s)
        elif key == "objslider":
            win.title("Objective Function Slider")
            win.geometry("760x640")
            self._viz_objslider(win, s)


    # ═══════════════════════════════════════════════════════════════════════════
    #  VIZ 1 — FEASIBLE REGION
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_fig(self, win, figsize=(7,5)):
        fig = Figure(figsize=figsize, facecolor=MPL_BG)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        return fig, canvas

    def _style_ax(self, ax, title):
        ax.set_facecolor(MPL_AX)
        ax.tick_params(colors=MPL_TEXT, labelsize=8)
        ax.xaxis.label.set_color(MPL_TEXT)
        ax.yaxis.label.set_color(MPL_TEXT)
        ax.title.set_color(ACCENT)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(MPL_GRID)
        ax.grid(True, color=MPL_GRID, linewidth=0.5, linestyle="--")
        ax.set_xlabel("x₁", fontsize=9)
        ax.set_ylabel("x₂", fontsize=9)

    def _get_plot_bounds(self, s):
        sol = s["sol"]
        x1_max = max(10.0, sol[0]*1.8) if sol[0] > 0 else 15.0
        x2_max = max(10.0, sol[1]*1.8) if sol[1] > 0 else 15.0
        return x1_max, x2_max

    def _draw_feasible_region(self, ax, s, alpha=0.18):
        """Shade the feasible region using a fine grid."""
        A_orig   = s["A_orig"]
        b_orig   = s["b_orig"]
        senses   = s["senses"]
        x1m, x2m = self._get_plot_bounds(s)

        x1g = np.linspace(0, x1m, 400)
        x2g = np.linspace(0, x2m, 400)
        X1, X2 = np.meshgrid(x1g, x2g)
        feasible = np.ones_like(X1, dtype=bool)

        for i in range(len(b_orig)):
            lhs = A_orig[i,0]*X1 + A_orig[i,1]*X2
            if senses[i] == "≤":
                feasible &= (lhs <= b_orig[i] + 1e-9)
            elif senses[i] == "≥":
                feasible &= (lhs >= b_orig[i] - 1e-9)
            else:
                feasible &= (np.abs(lhs - b_orig[i]) <= 1e-6*(abs(b_orig[i])+1))

        ax.contourf(X1, X2, feasible.astype(float),
                    levels=[0.5,1.5], colors=[SUCCESS], alpha=alpha)

    def _draw_constraints(self, ax, s):
        A_orig  = s["A_orig"]
        b_orig  = s["b_orig"]
        senses  = s["senses"]
        x1m, x2m = self._get_plot_bounds(s)
        x1g = np.linspace(0, x1m*1.1, 400)
        colors = [ACCENT, ACCENT2, RED, WARN, SUCCESS, "#c084fc"]

        for i in range(len(b_orig)):
            a1, a2 = A_orig[i,0], A_orig[i,1]
            bi = b_orig[i]
            col = colors[i % len(colors)]
            if abs(a2) > 1e-9:
                x2_line = (bi - a1*x1g) / a2
                mask = (x2_line >= -0.5) & (x2_line <= x2m*1.2)
                if mask.any():
                    ax.plot(x1g[mask], x2_line[mask], color=col,
                            linewidth=1.8, label=f"C{i+1}: {senses[i]} {bi:.4g}")
            elif abs(a1) > 1e-9:
                xv = bi / a1
                ax.axvline(xv, color=col, linewidth=1.8,
                           label=f"C{i+1}: {senses[i]} {bi:.4g}")

    def _viz_feasible(self, win, s):
        fig, canvas = self._make_fig(win, (7, 5.5))
        ax = fig.add_subplot(111)
        self._style_ax(ax, "Feasible Region")

        self._draw_feasible_region(ax, s)
        self._draw_constraints(ax, s)

        # optimal point
        sol = s["sol"]
        ax.scatter([sol[0]], [sol[1]], s=120, color=ACCENT2, zorder=5,
                   label=f"Optimal ({sol[0]:.3f}, {sol[1]:.3f})")
        ax.annotate(f"  ({sol[0]:.2f}, {sol[1]:.2f})\n  Z={s['z_orig']:.2f}",
                    (sol[0], sol[1]), color=ACCENT2, fontsize=8,
                    xytext=(sol[0]+0.3, sol[1]+0.3))

        x1m, x2m = self._get_plot_bounds(s)
        ax.set_xlim(0, x1m); ax.set_ylim(0, x2m)
        ax.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")
        fig.tight_layout()
        canvas.draw()

    # ═══════════════════════════════════════════════════════════════════════════
    #  VIZ 2 — SIMPLEX PATH
    # ═══════════════════════════════════════════════════════════════════════════

    def _viz_path(self, win, s):
        fig, canvas = self._make_fig(win, (7, 5.5))
        ax = fig.add_subplot(111)
        self._style_ax(ax, "Simplex Path across Iterations")

        self._draw_feasible_region(ax, s, alpha=0.12)
        self._draw_constraints(ax, s)

        # Extract x1, x2 at each iteration from basis/tableau history
        n = s["n"]
        path_pts = []
        for tableau, basis in zip(s["tableaux_history"], s["basis_history"]):
            sol_i = np.zeros(n + s["m"])
            for idx, bv in enumerate(basis):
                sol_i[bv] = tableau[idx, -1]
            path_pts.append((max(0, sol_i[0]), max(0, sol_i[1])))

        xs = [p[0] for p in path_pts]
        ys = [p[1] for p in path_pts]

        if len(xs) > 1:
            ax.plot(xs, ys, color=ACCENT2, linewidth=2,
                    linestyle="--", zorder=3, alpha=0.8)
            for i, (x, y) in enumerate(zip(xs, ys)):
                label = "Start" if i == 0 else f"Iter {i}"
                col   = SUCCESS if i == len(xs)-1 else ACCENT
                ax.scatter([x], [y], s=90, color=col, zorder=5)
                ax.annotate(f"  {label}\n  ({x:.2f},{y:.2f})",
                            (x, y), color=col, fontsize=7,
                            xytext=(x+0.2, y+0.2))
        else:
            ax.scatter(xs, ys, s=120, color=SUCCESS, zorder=5)
            ax.annotate("  Optimal\n  (already feasible)",
                        (xs[0], ys[0]), color=SUCCESS, fontsize=8)

        x1m, x2m = self._get_plot_bounds(s)
        ax.set_xlim(0, x1m); ax.set_ylim(0, x2m)
        ax.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")
        fig.tight_layout()
        canvas.draw()


    # ═══════════════════════════════════════════════════════════════════════════
    #  VIZ 3 — TABLEAU HEATMAP
    # ═══════════════════════════════════════════════════════════════════════════

    def _viz_heatmap(self, win, s):
        th = s["tableaux_history"]
        zh = s["z_row_history"]
        bh = s["basis_history"]
        n  = s["n"]
        m  = s["m"]

        # controls
        ctrl = tk.Frame(win, bg=MPL_BG)
        ctrl.pack(fill="x", padx=10, pady=6)
        tk.Label(ctrl, text="Iteration:", bg=MPL_BG, fg=TEXT,
                 font=("Consolas", 10)).pack(side="left", padx=6)
        itr_var = tk.IntVar(value=0)
        scale = tk.Scale(ctrl, from_=0, to=len(th)-1,
                         variable=itr_var, orient="horizontal",
                         bg=MPL_BG, fg=TEXT, highlightthickness=0,
                         troughcolor=PANEL2, activebackground=ACCENT,
                         length=300)
        scale.pack(side="left", padx=6)

        fig, canvas = self._make_fig(win, (8, 5))
        ax = fig.add_subplot(111)

        def draw(idx):
            ax.clear()
            tableau = th[idx]
            z_row   = zh[idx]
            basis   = bh[idx]
            total   = tableau.shape[1] - 1

            full = np.vstack([z_row[:-1], tableau[:, :-1]])
            im = ax.imshow(full, aspect="auto", cmap="RdYlGn",
                           interpolation="nearest")

            col_labels = [f"x{j+1}" if j < n else f"s{j-n+1}"
                          for j in range(total)]
            row_labels = ["Z"] + [f"x{bv+1}" if bv < n else f"s{bv-n+1}"
                                   for bv in basis]

            ax.set_xticks(range(total));  ax.set_xticklabels(col_labels, color=TEXT, fontsize=8)
            ax.set_yticks(range(m+1));    ax.set_yticklabels(row_labels, color=TEXT, fontsize=8)

            for r in range(m+1):
                for c2 in range(total):
                    val = full[r, c2]
                    ax.text(c2, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if -2<val<2 else "white")

            ttl = "Initial Tableau" if idx == 0 else f"After Iteration {idx}"
            ax.set_title(ttl, color=ACCENT, fontsize=11, fontweight="bold")
            ax.set_facecolor(MPL_AX)
            fig.tight_layout()
            canvas.draw()

        scale.config(command=lambda v: draw(int(v)))
        draw(0)

    # ═══════════════════════════════════════════════════════════════════════════
    #  VIZ 4 — CONSTRAINT INSPECTOR
    # ═══════════════════════════════════════════════════════════════════════════

    def _viz_inspector(self, win, s):
        A_orig = s["A_orig"]
        b_orig = s["b_orig"]
        senses = s["senses"]
        sol    = s["sol"]
        n      = s["n_orig"]
        m_orig = len(b_orig)

        tk.Label(win, text="CONSTRAINT INSPECTOR", bg=MPL_BG, fg=ACCENT,
                 font=("Consolas", 12, "bold")).pack(pady=(14,6))
        tk.Label(win, text="Check which constraints are binding at the optimal solution",
                 bg=MPL_BG, fg=SUBTEXT, font=("Consolas", 9)).pack(pady=(0,10))

        frame = tk.Frame(win, bg=MPL_BG)
        frame.pack(fill="both", expand=True, padx=20, pady=6)

        COLORS = [ACCENT, ACCENT2, RED, WARN, SUCCESS, "#c084fc"]
        for i in range(m_orig):
            a_row = A_orig[i]
            bi    = b_orig[i]
            sense = senses[i]
            lhs   = sum(a_row[j]*sol[j] for j in range(n))
            slack = lhs - bi
            binding = abs(slack) < 1e-6

            row_bg = "#1e2840" if i%2==0 else "#232640"
            rf = tk.Frame(frame, bg=row_bg, pady=6, padx=10,
                          highlightthickness=1, highlightbackground=BORDER)
            rf.pack(fill="x", pady=3)

            col = COLORS[i % len(COLORS)]
            tk.Label(rf, text=f"C{i+1}", bg=col, fg="#000",
                     font=("Consolas", 10, "bold"),
                     width=4, pady=4).pack(side="left", padx=(0,10))

            expr = " + ".join(f"{a_row[j]:.4g}·x{j+1}" for j in range(n))
            tk.Label(rf, text=f"{expr}  {sense}  {bi:.4g}",
                     bg=row_bg, fg=TEXT, font=("Consolas", 10)).pack(side="left")

            status_col  = SUCCESS if binding else ACCENT2
            status_text = "BINDING ✔" if binding else f"slack = {slack:.4f}"
            tk.Label(rf, text=status_text, bg=row_bg, fg=status_col,
                     font=("Consolas", 9, "bold")).pack(side="right", padx=10)

        # summary
        n_binding = sum(
            1 for i in range(m_orig)
            if abs(sum(A_orig[i,j]*sol[j] for j in range(n)) - b_orig[i]) < 1e-6
        )
        tk.Label(win,
                 text=f"{n_binding} of {m_orig} constraints are binding at the optimal point.",
                 bg=MPL_BG, fg=SUCCESS, font=("Consolas", 10, "bold")).pack(pady=12)

    # ═══════════════════════════════════════════════════════════════════════════
    #  VIZ 5 — OBJECTIVE SLIDER
    # ═══════════════════════════════════════════════════════════════════════════

    def _viz_objslider(self, win, s):
        c_orig  = s["c_orig"]
        obj_dir = s["obj_dir"]
        sol     = s["sol"]
        z_opt   = s["z_orig"]

        fig, canvas = self._make_fig(win, (7, 5))
        ax = fig.add_subplot(111)
        self._style_ax(ax, f"Objective Line — {obj_dir}  Z = {c_orig[0]:.4g}·x₁ + {c_orig[1]:.4g}·x₂")

        self._draw_feasible_region(ax, s, alpha=0.15)
        self._draw_constraints(ax, s)

        x1m, x2m = self._get_plot_bounds(s)
        ax.set_xlim(0, x1m); ax.set_ylim(0, x2m)

        x1g = np.linspace(0, x1m*1.1, 400)
        [obj_line] = ax.plot([], [], color=WARN, linewidth=2.5,
                             linestyle="-", label="Objective line", zorder=4)
        opt_pt = ax.scatter([sol[0]], [sol[1]], s=140, color=ACCENT2,
                            zorder=6, label=f"Optimal ({sol[0]:.2f},{sol[1]:.2f})")
        z_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                         color=WARN, fontsize=9, va="top",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL2, alpha=0.8))

        ax.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")
        fig.tight_layout()

        # slider
        ctrl = tk.Frame(win, bg=MPL_BG)
        ctrl.pack(fill="x", padx=10, pady=4)
        tk.Label(ctrl, text="Z value:", bg=MPL_BG, fg=TEXT,
                 font=("Consolas", 10)).pack(side="left", padx=6)

        z_min = min(0, z_opt*0.5) if z_opt > 0 else z_opt*1.5
        z_max = max(z_opt*2.0, z_opt+10)
        z_var = tk.DoubleVar(value=z_opt)
        scale = tk.Scale(ctrl, from_=z_min, to=z_max,
                         variable=z_var, orient="horizontal",
                         resolution=(z_max-z_min)/200,
                         bg=MPL_BG, fg=TEXT, highlightthickness=0,
                         troughcolor=PANEL2, activebackground=WARN,
                         length=400)
        scale.pack(side="left", padx=6)

        def update_obj(val):
            z = float(val)
            c1, c2 = c_orig[0], c_orig[1]
            if abs(c2) > 1e-9:
                x2_line = (z - c1*x1g) / c2
                mask = (x2_line >= 0) & (x2_line <= x2m*1.2)
                obj_line.set_data(x1g[mask], x2_line[mask])
            z_text.set_text(f"Z = {z:.2f}  (optimal = {z_opt:.2f})")
            canvas.draw_idle()

        scale.config(command=update_obj)
        update_obj(z_opt)


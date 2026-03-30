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

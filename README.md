# ⬡ Dual Simplex Method Solver

A desktop application built with Python and Tkinter that solves Linear Programming problems using the **Dual Simplex Method**, with full step-by-step tableau output and interactive visualizations.

> Mini Project — Operations Research | Information Technology (INFT-A)

---

## 📌 Features

- Solve LP problems with any number of variables and constraints (up to 10×10)
- Supports **Minimize** and **Maximize** objectives
- Supports **≤**, **≥**, and **=** constraint types
- Automatic conversion to standard form with detailed explanation
- Full **step-by-step tableau** display with pivot row/column highlighting
- Built-in **random example loader** for quick testing
- Interactive visualizations (for 2-variable problems):
  - 📐 Feasible Region
  - 🔀 Simplex Path across iterations
  - 🌡 Tableau Heatmap
  - 🔍 Constraint Inspector (binding vs. slack)
  - 🎚 Objective Function Slider

---

## 🖥️ Prerequisites

- **Python 3.8 or higher** — [Download here](https://www.python.org/downloads/)
- `tkinter` — comes **built-in** with Python on Windows and macOS
  - On **Linux**, install it separately:
    ```bash
    sudo apt install python3-tk
    ```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python main.py
```

> Replace `main.py` with the actual filename if different (e.g. `solver.py`).

---

## 🧭 How to Use

1. **Enter problem size** — set the number of variables and constraints
2. **Choose objective** — Minimize or Maximize
3. Click **▶ Generate Input Fields**
4. **Fill in** the objective coefficients, constraint matrix (A), RHS values (b), and constraint types
5. *(Optional)* Click **📋 Random Example** to auto-fill a sample problem
6. Click **⚡ Solve — Show All Steps** to run the solver
7. View the step-by-step solution in the output panel
8. After solving, use the **Visualization sidebar** (right side) to explore graphs and analysis

---

## 📦 Dependencies

| Package | Purpose |
|--------|---------|
| `numpy` | Matrix operations and tableau computation |
| `matplotlib` | Feasible region plots, heatmaps, and path visualizations |
| `tkinter` | GUI framework (built into Python) |

---

## 👥 Team

| Name | Roll No. |
|------|----------|
| Manthan Shah | 24101A0078 |
| Ishan Gurav | 24101A0077 |
| Prem Yelwande | 24101A0079 |
| Saukhya Gaikwad | 24101A0071 |
| Ayush Patil | 24101A0075 |
| Jiya Kanojia | 24101A0080 |

> Branch: INFT-A

---

## 📄 License

This project is submitted as a mini project for academic purposes.

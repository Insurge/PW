# nonlinear_regression_report.py
# ------------------------------
# pip install numpy pandas scipy matplotlib

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

def ensure_dir(path="outputs"):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---- моделі ----
def f_linear(x, a, b):
    return a + b*x

def f_quadratic(x, a, b, c):
    return a + b*x + c*x**2

def f_cubic(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

def f_power(x, a, b):
    # y = a * x^b , x>0
    return a * np.power(x, b)

def f_exponential(x, a, b):
    # y = a * exp(bx)
    return a * np.exp(b*x)

def f_hyperbola(x, a, b):
    # y = a + b/x
    return a + b / x

# ---- метрики ----
def r2_adj(y, yhat, k):
    n = len(y)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1-r2) * (n-1) / (n-k-1) if n > k+1 else np.nan
    return r2, adj, np.sqrt(ss_res/n), np.mean(np.abs(y - yhat))  # RMSE, MAE

def fit_model(x, y, kind):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # фільтр валідних точок (без нулів/від'ємних там, де потрібно)
    mask = np.isfinite(x) & np.isfinite(y)
    if kind in ["power", "hyperbola", "exponential"]:
        if kind in ["power", "hyperbola"]:
            mask &= x != 0
        if kind == "power" or kind == "exponential":
            mask &= x > 0
            if kind == "power":
                mask &= y > 0  # для степеневої
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None

    try:
        if kind == "linear":
            p0 = [np.median(y), 0.0]
            popt, _ = curve_fit(f_linear, x, y, p0=p0, maxfev=10000)
            yhat = f_linear(x, *popt)
            r2, adj, rmse, mae = r2_adj(y, yhat, k=1)
            return dict(kind=kind, params=popt, yhat=yhat, x=x, y=y, r2=r2, adj_r2=adj, rmse=rmse, mae=mae)

        if kind == "quadratic":
            p0 = [np.median(y), 0.0, 0.0]
            popt, _ = curve_fit(f_quadratic, x, y, p0=p0, maxfev=20000)
            yhat = f_quadratic(x, *popt)
            r2, adj, rmse, mae = r2_adj(y, yhat, k=2)
            return dict(kind=kind, params=popt, yhat=yhat, x=x, y=y, r2=r2, adj_r2=adj, rmse=rmse, mae=mae)

        if kind == "cubic":
            p0 = [np.median(y), 0.0, 0.0, 0.0]
            popt, _ = curve_fit(f_cubic, x, y, p0=p0, maxfev=50000)
            yhat = f_cubic(x, *popt)
            r2, adj, rmse, mae = r2_adj(y, yhat, k=3)
            return dict(kind=kind, params=popt, yhat=yhat, x=x, y=y, r2=r2, adj_r2=adj, rmse=rmse, mae=mae)

        if kind == "power":
            p0 = [np.median(y), 0.5]
            popt, _ = curve_fit(f_power, x, y, p0=p0, maxfev=20000)
            yhat = f_power(x, *popt)
            r2, adj, rmse, mae = r2_adj(y, yhat, k=2)
            return dict(kind=kind, params=popt, yhat=yhat, x=x, y=y, r2=r2, adj_r2=adj, rmse=rmse, mae=mae)

        if kind == "exponential":
            p0 = [np.median(y), 0.001]
            popt, _ = curve_fit(f_exponential, x, y, p0=p0, maxfev=20000)
            yhat = f_exponential(x, *popt)
            r2, adj, rmse, mae = r2_adj(y, yhat, k=2)
            return dict(kind=kind, params=popt, yhat=yhat, x=x, y=y, r2=r2, adj_r2=adj, rmse=rmse, mae=mae)

        if kind == "hyperbola":
            p0 = [np.median(y), 1.0]
            popt, _ = curve_fit(f_hyperbola, x, y, p0=p0, maxfev=20000)
            yhat = f_hyperbola(x, *popt)
            r2, adj, rmse, mae = r2_adj(y, yhat, k=2)
            return dict(kind=kind, params=popt, yhat=yhat, x=x, y=y, r2=r2, adj_r2=adj, rmse=rmse, mae=mae)
    except Exception:
        return None

def synthetic_data(case="sleep_productivity", n=150, seed=42):
    rng = np.random.default_rng(seed)
    if case == "sleep_productivity":
        x = rng.normal(7.0, 1.0, n).clip(3, 10)
        # параболічна залежність з максимумом ~7.5 год
        y = 70 + 8*(x - 7.5) - 1.2*(x - 7.5)**2 + rng.normal(0, 3, n)
        return x, y
    if case == "calories_weight":
        x = rng.normal(2300, 350, n).clip(1200, 3600)
        y = 50 + 0.02*(x-2000) - 0.000004*(x-2000)**2 + rng.normal(0, 2.5, n)
        return x, y
    # дефолт
    x = rng.uniform(1, 10, n)
    y = 3 + 2*np.sqrt(x) + rng.normal(0, 1, n)
    return x, y

def load_or_generate(csv, xcol, ycol, synth_case):
    if csv:
        df = pd.read_csv(csv)
        if xcol is None or ycol is None:
            cols = list(df.columns)
            ycol = cols[1] if ycol is None else ycol
            xcol = cols[0] if xcol is None else xcol
        x = df[xcol].values
        y = df[ycol].values
        name = f"CSV[{xcol}->{ycol}]"
    else:
        x, y = synthetic_data(synth_case)
        name = f"synthetic:{synth_case}"
    return x, y, name

def plot_models(outdir, x, y, results, title):
    # розкидані точки + криві
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 400)
    plt.figure()
    plt.scatter(x, y, s=12)
    for r in results:
        if r is None: 
            continue
        if r["kind"] == "linear":
            yh = f_linear(xs, *r["params"])
        elif r["kind"] == "quadratic":
            yh = f_quadratic(xs, *r["params"])
        elif r["kind"] == "cubic":
            yh = f_cubic(xs, *r["params"])
        elif r["kind"] == "power":
            yh = f_power(xs, *r["params"])
        elif r["kind"] == "exponential":
            yh = f_exponential(xs, *r["params"])
        elif r["kind"] == "hyperbola":
            yh = f_hyperbola(xs, *r["params"])
        else:
            continue
        plt.plot(xs, yh, label=f"{r['kind']} (Adj.R²={r['adj_r2']:.3f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    # Sanitize filename: remove invalid characters for Windows
    import re
    # First, replace colons with underscores for better separation
    safe_title = title.replace(":", "_")
    # Keep only alphanumeric, underscores, dots, hyphens, and spaces
    safe_title = "".join(c for c in safe_title if c.isalnum() or c in "._- ").strip()
    # Replace spaces with underscores
    safe_title = safe_title.replace(" ", "_")
    # Remove multiple consecutive underscores and clean up
    safe_title = re.sub(r"_+", "_", safe_title)
    safe_title = safe_title.strip("_")
    plt.savefig(outdir/f"plot_models_{safe_title}.png", bbox_inches="tight")
    plt.close()

def write_report(outdir, name, table, best_row):
    lines = []
    lines.append("# Комп’ютерний практикум №3 — Нелінійний регресійний аналіз\n")
    lines.append(f"**Дані:** {name}\n")
    lines.append("## Порівняння моделей")
    lines.append(table.to_csv(index=False))
    lines.append("## Найкраща модель")
    lines.append(f"- За Adj.R²: **{best_row['model']}**, Adj.R²={best_row['adj_r2']:.4f}, RMSE={best_row['rmse']:.4f}, MAE={best_row['mae']:.4f}\n")
    (outdir/f"report_draft_{name}.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Нелінійний регресійний аналіз (мульти-моделі)")
    ap.add_argument("--csv", type=str, default=None, help="Шлях до CSV")
    ap.add_argument("--x", type=str, default=None, help="Назва стовпця X")
    ap.add_argument("--y", type=str, default=None, help="Назва стовпця Y")
    ap.add_argument("--case", type=str, default="sleep_productivity", choices=["sleep_productivity","calories_weight","custom"], help="Синтетичний кейс (якщо немає CSV)")
    ap.add_argument("--outputs", type=str, default="outputs", help="Папка для результатів")
    args = ap.parse_args()

    outdir = ensure_dir(args.outputs)
    x, y, name = load_or_generate(args.csv, args.x, args.y, args.case)

    models = ["linear", "quadratic", "cubic", "power", "exponential", "hyperbola"]
    results = [fit_model(x, y, m) for m in models]
    results = [r for r in results if r is not None]

    rows = []
    for r in results:
        rows.append(dict(
            model=r["kind"],
            a=r["params"][0],
            b=(r["params"][1] if len(r["params"])>1 else np.nan),
            c=(r["params"][2] if len(r["params"])>2 else np.nan),
            d=(r["params"][3] if len(r["params"])>3 else np.nan),
            r2=r["r2"], adj_r2=r["adj_r2"], rmse=r["rmse"], mae=r["mae"]
        ))
    table = pd.DataFrame(rows).sort_values(["adj_r2","r2"], ascending=[False, False])
    # Sanitize name for filenames - better word separation
    import re
    # First, replace colons with underscores for better separation
    safe_name = name.replace(":", "_")
    # Keep only alphanumeric, underscores, dots, hyphens, and spaces
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ").strip()
    # Replace spaces with underscores
    safe_name = safe_name.replace(" ", "_")
    # Remove multiple consecutive underscores and clean up
    safe_name = re.sub(r"_+", "_", safe_name)
    safe_name = safe_name.strip("_")
    
    table.to_csv(outdir/f"models_comparison_{safe_name}.csv", index=False)

    if not table.empty:
        best_row = table.iloc[0]
        write_report(outdir, safe_name, table, best_row)

    plot_models(outdir, x, y, results, safe_name)

    print("Збережено у:", outdir.resolve())
    print(f" - models_comparison_{safe_name}.csv")
    print(f" - plot_models_{safe_name}.png")
    print(f" - report_draft_{safe_name}.md")

if __name__ == "__main__":
    main()

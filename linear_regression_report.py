# linear_regression_report.py
# ---------------------------
# Вимоги: pandas, numpy, scipy, statsmodels, matplotlib
# pip install pandas numpy scipy statsmodels matplotlib

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
# statsmodels is not available on some Python builds; we implement OLS/VIF/diagnostics manually

import matplotlib.pyplot as plt

from typing import Tuple, Dict

def ensure_outputs_dir(path="outputs"):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def load_or_make_data(csv_path, y_name, x_names, n=200, random_state=42):
    if csv_path:
        df = pd.read_csv(csv_path)
        # Якщо користувач не задав імена змінних — візьмемо перший стовпець як y, решту як X
        if y_name is None:
            y_name = df.columns[0]
        if not x_names:
            x_names = [c for c in df.columns if c != y_name]
        return df, y_name, x_names

    # Синтетика: правдоподібні зв’язки + шум
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame({
        "sleep_hours": rng.normal(7.2, 1.0, n).clip(3.0, 10.0),
        "steps_per_day": rng.normal(8000, 2500, n).clip(500, 20000),
        "calories_intake": rng.normal(2300, 350, n).clip(1200, 3500),
        "weight_kg": rng.normal(80, 12, n).clip(45, 140),
        "caffeine_mg": rng.normal(120, 60, n).clip(0, 500),
    })
    # Залежна: продуктивність залежить від сну (+), кроків (+ слабко), калорій (незначно), ваги (- слабко), кофеїну (+/0)
    eps = rng.normal(0, 5, n)
    df["productivity_score"] = (
        6.0*df["sleep_hours"]
        + 0.0003*df["steps_per_day"]
        + 0.002*df["caffeine_mg"]
        - 0.10*df["weight_kg"]
        + 0.001*df["calories_intake"]
        + eps + 40
    ).clip(0, 100)

    # Стандартні за замовчуванням змінні для моделі
    y_name = "productivity_score" if y_name is None else y_name
    if not x_names:
        x_names = ["sleep_hours", "steps_per_day", "calories_intake", "weight_kg", "caffeine_mg"]
    return df, y_name, x_names

def make_design_matrices(df, y_name, x_names):
    # one-hot для категоріальних
    df_model = df[[y_name] + x_names].copy()
    df_model = pd.get_dummies(df_model, drop_first=True)
    # listwise deletion пропусків
    df_model = df_model.dropna()

    y = df_model[y_name].astype(float)
    X = df_model.drop(columns=[y_name])
    # Додаємо константу як стовпець 'const'
    X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)

    return df_model, y, X

def fit_ols(y: pd.Series, X: pd.DataFrame, robust: bool = False) -> Dict:
    # Аналітичне розв'язання OLS: beta = (X'X)^{-1} X'y
    X_mat = X.values.astype(float)
    y_vec = y.values.astype(float)

    XtX = X_mat.T @ X_mat
    # Використовуємо псевдоінверсію для стійкості
    XtX_inv = np.linalg.pinv(XtX)
    Xty = X_mat.T @ y_vec
    beta = XtX_inv @ Xty

    fitted = X_mat @ beta
    resid = y_vec - fitted

    n = X_mat.shape[0]
    p = X_mat.shape[1]

    rss = float(resid.T @ resid)
    tss = float(((y_vec - y_vec.mean()) ** 2).sum())
    rsq = 1.0 - (rss / tss) if tss > 0 else np.nan
    rsq_adj = 1.0 - (1.0 - rsq) * (n - 1) / (n - p) if n > p else np.nan

    df_resid = n - p
    sigma2 = rss / df_resid if df_resid > 0 else np.nan
    # Коваріаційна матриця: класична або робастна (HC0)
    if robust and df_resid > 0:
        W = np.diag(resid ** 2)
        cov_beta = XtX_inv @ (X_mat.T @ W @ X_mat) @ XtX_inv
    else:
        cov_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(cov_beta))

    # t-статистики та p-значення
    with np.errstate(divide='ignore', invalid='ignore'):
        t_vals = beta / se_beta
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_vals), df=df_resid)) if df_resid > 0 else np.full_like(beta, np.nan)

    # 95% ДІ
    alpha = 0.05
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df_resid) if df_resid > 0 else np.nan
    ci_low = beta - t_crit * se_beta if df_resid > 0 else np.full_like(beta, np.nan)
    ci_high = beta + t_crit * se_beta if df_resid > 0 else np.full_like(beta, np.nan)

    # F-тест на значущість моделі (без константи): k = p-1
    k = max(p - 1, 0)
    if k > 0 and n - p > 0:
        f_stat = (rsq / k) / ((1.0 - rsq) / (n - p)) if 0 < rsq < 1 else np.nan
        f_pvalue = stats.f.sf(f_stat, k, n - p) if not np.isnan(f_stat) else np.nan
    else:
        f_stat = np.nan
        f_pvalue = np.nan

    coef_names = list(X.columns)
    coef_df = pd.DataFrame({
        "term": coef_names,
        "coef": beta,
        "std_err": se_beta,
        "t": t_vals,
        "p_value": p_vals,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })

    return {
        "coefficients": coef_df,
        "nobs": n,
        "rsquared": rsq,
        "rsquared_adj": rsq_adj,
        "fvalue": f_stat,
        "f_pvalue": f_pvalue,
        "fittedvalues": fitted,
        "resid": resid,
        "X": X,
        "sigma2": sigma2,
        "robust": robust,
    }

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    # Рахуємо VIF вручну: VIF_j = 1 / (1 - R_j^2), де R_j^2 — регресія X_j на інші X
    cols = [c for c in X.columns if c != "const"]
    vif_rows = []
    for col in cols:
        others = [c for c in cols if c != col]
        # додаємо константу для допоміжної регресії
        X_aux = pd.concat([pd.Series(1.0, index=X.index, name="const"), X[others]], axis=1).values
        y_aux = X[col].values
        # OLS для допоміжної регресії
        beta_aux = np.linalg.lstsq(X_aux, y_aux, rcond=None)[0]
        fitted_aux = X_aux @ beta_aux
        ss_res = float(((y_aux - fitted_aux) ** 2).sum())
        ss_tot = float(((y_aux - y_aux.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif_val = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_rows.append((col, float(vif_val)))
    return pd.DataFrame(vif_rows, columns=["variable", "VIF"])

def residual_diagnostics(res: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
    resid = res["resid"]
    fitted = res["fittedvalues"]

    # Shapiro–Wilk (для n<=5000)
    sw_stat, sw_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)

    # Durbin–Watson вручну
    diff_resid = np.diff(resid)
    dw = float((diff_resid @ diff_resid) / (resid @ resid)) if float(resid @ resid) > 0 else np.nan

    # Breusch–Pagan: aux регресія (e^2 / sigma2) на X (включаючи константу). LM = n * R^2, df = k (без константи)
    sigma2 = res.get("sigma2", np.nan)
    e2 = (resid ** 2) / sigma2 if (isinstance(sigma2, (int, float)) and sigma2 > 0) else (resid ** 2)
    X_aux = X.values.astype(float)
    beta_aux = np.linalg.lstsq(X_aux, e2, rcond=None)[0]
    fitted_aux = X_aux @ beta_aux
    ss_res = float(((e2 - fitted_aux) ** 2).sum())
    ss_tot = float(((e2 - e2.mean()) ** 2).sum())
    r2_aux = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = X.shape[0]
    k = X.shape[1] - 1  # без константи
    lm = n * r2_aux
    bp_p = 1.0 - stats.chi2.cdf(lm, df=k) if k > 0 else np.nan

    return {
        "shapiro_stat": sw_stat, "shapiro_p": sw_p,
        "durbin_watson": dw,
        "breusch_pagan_stat": lm, "breusch_pagan_p": bp_p,
        "resid": resid, "fitted": fitted
    }

def save_tables(outputs_dir, res: Dict, vif_df, diag, df_model, y_name, x_names):
    # Коефіцієнти з CI
    coef_df = res["coefficients"].copy()
    coef_df.to_csv(outputs_dir/"coefficients.csv", index=False)

    # Узагальнені метрики
    metrics = pd.DataFrame({
        "n_obs": [int(res["nobs"])],
        "r_squared": [res["rsquared"]],
        "adj_r_squared": [res["rsquared_adj"]],
        "f_stat": [res["fvalue"]],
        "f_p_value": [res["f_pvalue"]]
    })
    metrics.to_csv(outputs_dir/"model_metrics.csv", index=False)

    # VIF
    vif_df.to_csv(outputs_dir/"vif.csv", index=False)

    # Залишки
    resid = res["resid"]
    fitted = res["fittedvalues"]
    pd.DataFrame({
        "fitted": fitted,
        "residuals": resid,
        "standardized_residuals": (resid - np.mean(resid)) / np.std(resid, ddof=1)
    }).to_csv(outputs_dir/"residuals.csv", index=False)

    # Текстовий summary
    with open(outputs_dir/"ols_summary.txt", "w", encoding="utf-8") as f:
        f.write(
            "OLS Summary (custom)\n"
            f"n={res['nobs']}, p={res['X'].shape[1]}\n"
            f"R^2={res['rsquared']:.6f}, Adj. R^2={res['rsquared_adj']:.6f}\n"
            f"F={res['fvalue'] if not np.isnan(res['fvalue']) else 'nan'}"
            f" (p={res['f_pvalue'] if not np.isnan(res['f_pvalue']) else 'nan'})\n\n"
        )
        f.write("Coefficients:\n")
        f.write(coef_df.to_string(index=False))

def save_plots(outputs_dir, diag, y, X, y_name):
    # 1) y vs fitted
    plt.figure()
    plt.scatter(diag["fitted"], y)
    plt.xlabel("Fitted values")
    plt.ylabel(f"Observed {y_name}")
    plt.title("Observed vs Fitted")
    plt.savefig(outputs_dir/"plot_observed_vs_fitted.png", bbox_inches="tight")
    plt.close()

    # 2) Residuals vs fitted
    plt.figure()
    plt.scatter(diag["fitted"], diag["resid"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.savefig(outputs_dir/"plot_residuals_vs_fitted.png", bbox_inches="tight")
    plt.close()

    # 3) Histogram of residuals
    plt.figure()
    plt.hist(diag["resid"], bins=20)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.savefig(outputs_dir/"plot_residuals_hist.png", bbox_inches="tight")
    plt.close()

    # 4) QQ-plot
    plt.figure()
    stats.probplot(diag["resid"], dist="norm", plot=plt)
    plt.title("QQ-plot of Residuals")
    plt.savefig(outputs_dir/"plot_residuals_qq.png", bbox_inches="tight")
    plt.close()

def write_markdown_report(outputs_dir, csv_path, y_name, x_names, res, diag):
    md = []
    md.append("# Комп’ютерний практикум №2 — Лінійний регресійний аналіз\n")
    md.append("**Мета:** побудувати лінійну регресійну модель та оцінити її якість.\n")
    md.append("## Дані")
    md.append(f"- Джерело: {'CSV: ' + csv_path if csv_path else 'Синтетично згенеровані'}")
    md.append(f"- Залежна змінна (Y): `{y_name}`")
    md.append(f"- Незалежні змінні (X): `{', '.join(x_names)}`\n")

    md.append("## Результати моделювання (OLS)")
    md.append(f"- n={int(res['nobs'])}  \n- R²={res['rsquared']:.4f}  \n- Adj.R²={res['rsquared_adj']:.4f}  \n- F={res['fvalue']:.3f} (p={res['f_pvalue']:.3g})\n")

    sw = diag["shapiro_p"]
    bp = diag["breusch_pagan_p"]
    md.append("## Діагностика залишків")
    md.append(f"- Shapiro–Wilk p={sw:.3g} (нормальність залишків {'не відхиляється' if (not np.isnan(sw) and sw>=0.05) else 'відхиляється'})")
    md.append(f"- Durbin–Watson={diag['durbin_watson']:.3f} (≈2 бажано)")
    md.append(f"- Breusch–Pagan p={bp:.3g} (гомоскедастичність {'ок' if bp>=0.05 else 'порушена'})\n")

    md.append("## Графіки")
    md.append("- `plot_observed_vs_fitted.png` — спостережені значення vs передбачені")
    md.append("- `plot_residuals_vs_fitted.png` — залишки vs передбачені")
    md.append("- `plot_residuals_hist.png` — гістограма залишків")
    md.append("- `plot_residuals_qq.png` — QQ-plot залишків\n")

    md.append("## Висновки (шаблон)")
    md.append("- Модель статистично значуща за F-тестом; кількість поясненої варіації відображена R²/Adj.R².")
    md.append("- Перевірити важливість окремих предикторів за p-значеннями коефіцієнтів і 95% ДІ (див. `coefficients.csv`).")
    md.append("- Переконатися, що VIF < 10 (мультиколінеарність не критична).")
    md.append("- Оцінити припущення нормальності/гомоскедастичності за діагностикою та графіками.")
    md.append("")

    (outputs_dir/"report_draft.md").write_text("\n".join(md), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Лінійний регресійний аналіз (OLS) з автозвітом.")
    parser.add_argument("--csv", type=str, default=None, help="Шлях до CSV з даними.")
    parser.add_argument("--y", type=str, default=None, help="Ім'я залежної змінної (Y).")
    parser.add_argument("--X", nargs="*", default=None, help="Список незалежних змінних (X). Якщо не задано — всі, окрім Y.")
    parser.add_argument("--outputs", type=str, default="outputs", help="Папка для результатів (за замовчуванням 'outputs').")
    parser.add_argument("--robust", action="store_true", help="Робастні стандартні похибки (HC0).")
    args = parser.parse_args()

    out_dir = ensure_outputs_dir(args.outputs)

    df, y_name, x_names = load_or_make_data(args.csv, args.y, args.X)
    df_model, y, X = make_design_matrices(df, y_name, x_names)

    res = fit_ols(y, X, robust=args.robust)
    vif_df = compute_vif(X)

    diag = residual_diagnostics(res, X, y)
    save_tables(out_dir, res, vif_df, diag, df_model, y_name, x_names)
    save_plots(out_dir, diag, y, X, y_name)
    write_markdown_report(out_dir, args.csv, y_name, x_names, res, diag)

    print("\n=== Готово! Результати збережено у:", out_dir.resolve(), "===\n")
    print("Файли:")
    print(" - ols_summary.txt")
    print(" - coefficients.csv")
    print(" - model_metrics.csv")
    print(" - vif.csv")
    print(" - residuals.csv")
    print(" - report_draft.md")
    print(" - plot_observed_vs_fitted.png")
    print(" - plot_residuals_vs_fitted.png")
    print(" - plot_residuals_hist.png")
    print(" - plot_residuals_qq.png")

if __name__ == "__main__":
    main()

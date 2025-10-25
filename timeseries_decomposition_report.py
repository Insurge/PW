# timeseries_decomposition_report.py
# ---------------------------------
# Вимоги: pandas, numpy, scipy, matplotlib
# pip install pandas numpy scipy matplotlib

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

def ensure_dir(p="outputs"):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

def load_or_make(csv_path: str | None, n: int = 120, seed: int = 42):
    if csv_path:
        df = pd.read_csv(csv_path)
        # очікуємо колонки: date, value
        if "date" not in df.columns or "value" not in df.columns:
            raise ValueError("CSV має містити колонки 'date' і 'value'")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.set_index("date")
        return df[["value"]].astype(float), "CSV"
    # --- синтетика: місячний ряд із трендом + сезонністю + шумом ---
    rng = np.random.default_rng(seed)
    t = pd.date_range("2015-01-01", periods=n, freq="MS")
    trend = 0.3*np.arange(n)     # слабкий зростальний тренд
    season = 5*np.sin(2*np.pi*np.arange(n)/12)   # сезонність 12
    noise = rng.normal(0, 2, n)
    y = 50 + trend + season + noise
    df = pd.DataFrame({"value": y}, index=t)
    return df, "synthetic_monthly"

def moving_average(x: pd.Series, window: int = 12) -> pd.Series:
    return x.rolling(window=window, center=True, min_periods=max(1, window//2)).mean()

def simple_decompose(y: pd.Series, period: int = 12):
    """Проста декомпозиція без statsmodels"""
    # Тренд через ковзне середнє
    trend = moving_average(y, window=period)
    
    # Залишок після видалення тренду
    detrended = y - trend
    
    # Сезонність - середнє за кожен період
    n_periods = len(y) // period
    seasonal_pattern = np.zeros(period)
    for i in range(period):
        idx = np.arange(i, len(y), period)[:n_periods]
        if len(idx) > 0:
            seasonal_pattern[i] = detrended.iloc[idx].mean()
    
    # Розширюємо до довжини ряду
    seasonal = np.tile(seasonal_pattern, n_periods + 1)[:len(y)]
    seasonal = pd.Series(seasonal, index=y.index)
    
    # Залишки
    resid = detrended - seasonal
    
    return trend, seasonal, resid

def plot_acf_simple(resid, lags=50, ax=None):
    """Простий ACF без statsmodels"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))
    
    resid_clean = resid.dropna()
    n = len(resid_clean)
    max_lags = min(lags, n-2)
    
    autocorr = []
    for lag in range(max_lags + 1):
        corr = resid_clean.autocorr(lag=lag)
        autocorr.append(corr)
    
    ax.bar(range(len(autocorr)), autocorr, width=0.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhspan(-1.96/np.sqrt(n), 1.96/np.sqrt(n), alpha=0.2, color='gray')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('ACF of irregular component')
    ax.grid(True, alpha=0.3)
    
    return ax

def cox_stuart_trend_test(x: np.ndarray):
    """
    Cox–Stuart test: розбиваємо ряд на дві половини й порівнюємо попарно.
    H0: тренду немає (симетрія знаків різниць)
    """
    n = len(x)
    half = n//2
    a = x[:half]
    b = x[n-half:]  # якщо непарне n — середню точку ігноруємо
    d = b - a
    # відкидаємо нульові різниці
    d = d[d != 0]
    if len(d) == 0:
        # усе рівне — трактуємо як відсутність тренду
        return {"n_pairs": 0, "n_pos": 0, "n_neg": 0, "p_value": 1.0, "direction": "none"}
    n_pos = np.sum(d > 0)
    n_neg = np.sum(d < 0)
    k = min(n_pos, n_neg)
    n_eff = n_pos + n_neg
    # двостороння біноміальна перевірка (ймовірність 0.5)
    p = binomtest(k, n_eff, 0.5, alternative="two-sided").pvalue
    direction = "up" if n_pos > n_neg else "down"
    return {"n_pairs": int(n_eff), "n_pos": int(n_pos), "n_neg": int(n_neg), "p_value": float(p), "direction": direction}

def main():
    ap = argparse.ArgumentParser(description="Часові ряди: тренд, серійний критерій, декомпозиція, ACF залишків")
    ap.add_argument("--csv", type=str, default=None, help="Шлях до CSV із колонками date,value")
    ap.add_argument("--period", type=int, default=12, help="Сезонний період (приклад: 12 для місячних)")
    ap.add_argument("--model", type=str, default="additive", choices=["additive","multiplicative"], help="Модель декомпозиції")
    ap.add_argument("--outputs", type=str, default="outputs")
    args = ap.parse_args()

    out = ensure_dir(args.outputs)
    df, source = load_or_make(args.csv)
    y = df["value"].astype(float)

    # 1) Графік ряду + ковзне середнє
    plt.figure(figsize=(10,4))
    plt.plot(y.index, y.values, label="value")
    ma = moving_average(y, window=args.period)
    plt.plot(ma.index, ma.values, label=f"Moving average ({args.period})")
    plt.title(f"Time series ({source})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/"01_series_and_ma.png")
    plt.close()

    # 2) Критерій «висхідних і низхідних» серій (Cox–Stuart)
    cs = cox_stuart_trend_test(y.values.copy())
    trend_comment = "тенденція відсутня"
    if cs["p_value"] < 0.05:
        trend_comment = f"тенденція присутня ({'зростаюча' if cs['direction']=='up' else 'спадна'}), p={cs['p_value']:.3g}"
    else:
        trend_comment = f"тенденція статистично не підтверджена (p={cs['p_value']:.3g})"

    # 3) Декомпозиція
    trend, seasonal, resid = simple_decompose(y, period=args.period)
    comp = pd.DataFrame({
        "observed": y,
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid
    })
    comp.to_csv(out/"02_components.csv", index=True)

    # 4) Графіки компонент
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(y.index, y.values)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Time Series Decomposition')
    
    axes[1].plot(trend.index, trend.values)
    axes[1].set_ylabel('Trend')
    
    axes[2].plot(seasonal.index, seasonal.values)
    axes[2].set_ylabel('Seasonal')
    
    axes[3].plot(resid.index, resid.values)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(out/"02_decomposition.png")
    plt.close()

    # 5) ACF залишків (irregular component)
    resid = comp["resid"].dropna()
    fig, ax = plt.subplots(figsize=(8,4))
    plot_acf_simple(resid, lags=min(50, len(resid)-2), ax=ax)
    plt.tight_layout()
    plt.savefig(out/"03_acf_irregular.png")
    plt.close()

    # 6) Короткий звіт (markdown)
    md = []
    md.append("# Практична робота №1 — Аналіз часових рядів (ЗМІ)\n")
    md.append(f"**Дані:** {source}; **модель:** {args.model}; **період сезонності:** {args.period}\n")
    md.append("## Крок 1–3. Візуалізація та попередній висновок")
    md.append("- Файл `01_series_and_ma.png`: початковий ряд та ковзне середнє.")
    md.append(f"- Попередній висновок про тренд: {trend_comment}\n")
    md.append("## Крок 4. Критерій «висхідних і низхідних» серій (Cox–Stuart)")
    md.append(f"- Пар порівнянь: {cs['n_pairs']}, позитивних різниць: {cs['n_pos']}, негативних: {cs['n_neg']}.")
    md.append(f"- p-value = {cs['p_value']:.5f}\n")
    md.append("## Крок 5. Декомпозиція")
    md.append("- Файл `02_decomposition.png` та таблиця `02_components.csv` (observed, trend, seasonal, resid).")
    md.append("- Інтерпретація: тренд — довгострокова складова; seasonal — періодична; resid — шум.\n")
    md.append("## Крок 6. ACF залишків")
    md.append("- Файл `03_acf_irregular.png`.")
    md.append("- Якщо більшість значень ACF залишків лежать у довірчих межах і швидко спадають до нуля — розкладення коректне (залишки ≈ «білий шум»).\n")
    (out/"report_draft.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== Готово! Артефакти у:", out.resolve(), "===\n")
    print(" - 01_series_and_ma.png")
    print(" - 02_decomposition.png")
    print(" - 02_components.csv")
    print(" - 03_acf_irregular.png")
    print(" - report_draft.md")

if __name__ == "__main__":
    main()

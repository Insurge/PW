#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комп'ютерний практикум №1: Метод кореляційного аналізу даних
Виконання всього аналізу одним скриптом
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, kstest, pearsonr, spearmanr
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Параметри аналізу
user_data_path = None  # або 'path/to/data.csv'
alpha = 0.05
random_state = 42

# Встановлення seed для відтворюваності
np.random.seed(random_state)

# Створення папки для результатів
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("ПОЧАТОК КОРЕЛЯЦІЙНОГО АНАЛІЗУ")
print("=" * 80)
print("[OK] Всі бібліотеки імпортовано")
print(f"[OK] Папка для результатів: {output_dir.absolute()}")
print()

# Функція генерації даних
def generate_synthetic_data(n_samples=200, missing_rate=0.07):
    """Генерація синтетичних даних з правдоподібними кореляціями"""
    np.random.seed(random_state)
    
    # Базові незалежні змінні
    sleep_hours = np.random.normal(7, 1.2, n_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    
    steps_per_day = np.random.normal(8000, 2500, n_samples)
    steps_per_day = np.clip(steps_per_day, 2000, 15000)
    
    caffeine_mg = np.random.normal(150, 80, n_samples)
    caffeine_mg = np.clip(caffeine_mg, 0, 400)
    
    calories_intake = np.random.normal(2000, 400, n_samples)
    calories_intake = np.clip(calories_intake, 1200, 3500)
    
    # Залежні змінні з кореляціями
    productivity_score = (
        30 + 
        6 * sleep_hours +
        0.03 * caffeine_mg +
        np.random.normal(0, 8, n_samples)
    )
    productivity_score = np.clip(productivity_score, 0, 100)
    
    weight_kg = (
        40 + 
        0.012 * calories_intake +
        -0.0015 * steps_per_day +
        np.random.normal(0, 5, n_samples)
    )
    weight_kg = np.clip(weight_kg, 50, 100)
    
    df = pd.DataFrame({
        'sleep_hours': sleep_hours,
        'productivity_score': productivity_score,
        'steps_per_day': steps_per_day,
        'calories_intake': calories_intake,
        'weight_kg': weight_kg,
        'caffeine_mg': caffeine_mg
    })
    
    # Додавання пропусків
    n_missing = int(n_samples * len(df.columns) * missing_rate)
    missing_indices = np.random.choice(
        n_samples * len(df.columns), 
        size=n_missing, 
        replace=False
    )
    
    for idx in missing_indices:
        row = idx // len(df.columns)
        col = idx % len(df.columns)
        df.iloc[row, col] = np.nan
    
    return df

# Генерація даних
print("Крок 1: Генерація даних")
print("-" * 80)
if user_data_path is not None:
    df = pd.read_csv(user_data_path)
    print(f"[OK] Дані завантажено з: {user_data_path}")
else:
    df = generate_synthetic_data(n_samples=200, missing_rate=0.07)
    synthetic_path = output_dir / 'synthetic_data.csv'
    df.to_csv(synthetic_path, index=False)
    print(f"[OK] Синтетичні дані згенеровано")
    print(f"[OK] Збережено: {synthetic_path}")

print(f"[OK] Розмір датасету: {df.shape[0]} рядків x {df.shape[1]} стовпців")
print(f"\nПерші 5 рядків:")
print(df.head())
print()

# Аналіз пропусків
print("Крок 2: Аналіз пропущених значень")
print("-" * 80)
missing_data = pd.DataFrame({
    'Змінна': df.columns,
    'Пропусків': df.isnull().sum(),
    'Частка (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_data.to_string(index=False))
print(f"\nЗагальна частка пропусків: {df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%")
print()

# Обробка пропусків
print("Крок 3: Обробка пропусків")
print("-" * 80)
df_pairwise = df.copy()
df_listwise = df.dropna()
print(f"[OK] Pairwise: {len(df_pairwise)} спостережень")
print(f"[OK] Listwise: {len(df_listwise)} спостережень (-{len(df) - len(df_listwise)} рядків)")
print()

# Тести нормальності
print("Крок 4: Перевірка нормальності")
print("-" * 80)

def test_normality(data, var_name, alpha=0.05):
    """Перевірка нормальності розподілу"""
    clean_data = data.dropna()
    n = len(clean_data)
    
    if n == 0:
        return {
            'variable': var_name,
            'test': 'N/A',
            'statistic': np.nan,
            'p_value': np.nan,
            'n': 0,
            'normality_conclusion': 'Недостатньо даних',
            'is_normal': False
        }
    
    if n <= 5000:
        test_name = 'Shapiro-Wilk'
        statistic, p_value = shapiro(clean_data)
    else:
        test_name = 'Kolmogorov-Smirnov'
        mean, std = clean_data.mean(), clean_data.std()
        statistic, p_value = kstest(clean_data, lambda x: stats.norm.cdf(x, mean, std))
    
    if p_value >= alpha:
        conclusion = f'Нормальний (p={p_value:.4f} >= {alpha})'
        is_normal = True
    else:
        conclusion = f'Не нормальний (p={p_value:.4f} < {alpha})'
        is_normal = False
    
    return {
        'variable': var_name,
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'n': n,
        'normality_conclusion': conclusion,
        'is_normal': is_normal
    }

normality_results = []
for col in df_listwise.select_dtypes(include=[np.number]).columns:
    result = test_normality(df_listwise[col], col, alpha)
    normality_results.append(result)

normality_df = pd.DataFrame(normality_results)
normality_df_display = normality_df[['variable', 'test', 'statistic', 'p_value', 'n', 'normality_conclusion']]

normality_path = output_dir / 'normality_summary.csv'
normality_df_display.to_csv(normality_path, index=False)

print(normality_df_display.to_string(index=False))
print(f"\n[OK] Результати збережено: {normality_path}")
print()

normality_dict = {row['variable']: row['is_normal'] for _, row in normality_df.iterrows()}

# Кореляційний аналіз
print("Крок 5: Розрахунок кореляцій")
print("-" * 80)

def interpret_strength(r):
    """Інтерпретація сили зв'язку"""
    abs_r = abs(r)
    direction = "позитивний" if r >= 0 else "негативний"
    
    if abs_r >= 0.75:
        strength = "дуже високий"
    elif abs_r >= 0.50:
        strength = "високий"
    elif abs_r >= 0.25:
        strength = "середній"
    else:
        strength = "слабкий"
    
    return f"{strength} {direction}"

def choose_correlation_method(x, y, var_x, var_y, normality_dict):
    """Вибір методу кореляції"""
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)
    
    if n < 3:
        return 'N/A', np.nan, np.nan, n
    
    x_normal = normality_dict.get(var_x, False)
    y_normal = normality_dict.get(var_y, False)
    
    if x_normal and y_normal:
        method = 'Pearson'
        coef, p_val = pearsonr(x_clean, y_clean)
    else:
        method = 'Spearman'
        coef, p_val = spearmanr(x_clean, y_clean)
    
    return method, coef, p_val, n

def calculate_all_correlations(df, normality_dict, approach_name, alpha=0.05):
    """Обчислення кореляцій для всіх пар"""
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for i, var_x in enumerate(numeric_cols):
        for var_y in numeric_cols[i+1:]:
            method, coef, p_val, n = choose_correlation_method(
                df[var_x], df[var_y], var_x, var_y, normality_dict
            )
            
            if not np.isnan(coef):
                if p_val < 0.01:
                    significance = '**'
                elif p_val < alpha:
                    significance = '*'
                else:
                    significance = ''
                
                h0_conclusion = 'Відхилено' if p_val < alpha else 'Не відхилено'
                
                results.append({
                    'variable_x': var_x,
                    'variable_y': var_y,
                    'method': method,
                    'coefficient': coef,
                    'p_value': p_val,
                    'n': n,
                    'significance': significance,
                    'strength_label': interpret_strength(coef),
                    'H0_conclusion': h0_conclusion,
                    'approach': approach_name
                })
    
    return pd.DataFrame(results)

# Обчислення кореляцій
corr_listwise = calculate_all_correlations(df_listwise, normality_dict, 'listwise', alpha)
corr_listwise_path = output_dir / 'correlations_listwise.csv'
corr_listwise.to_csv(corr_listwise_path, index=False)

print("[OK] Кореляції розраховано (listwise)")
print(f"[OK] Збережено: {corr_listwise_path}")
print(f"\nТоп-5 найсильніших кореляцій:")
top_corr = corr_listwise.nlargest(5, 'coefficient')[['variable_x', 'variable_y', 'method', 'coefficient', 'p_value', 'strength_label']]
print(top_corr.to_string(index=False))
print()

# Візуалізація
print("Крок 6: Створення графіків")
print("-" * 80)

def create_scatter_plot(df, x_var, y_var, method, r, p_val, n, output_path):
    """Створення діаграми розсіювання"""
    mask = ~(pd.isna(df[x_var]) | pd.isna(df[y_var]))
    x_clean = df[x_var][mask]
    y_clean = df[y_var][mask]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_clean, y_clean)
    
    if method == 'Pearson':
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        plt.plot(x_clean, p(x_clean), linestyle='--')
    
    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel(y_var.replace('_', ' ').title())
    plt.title(f'{x_var} vs {y_var}\n{method}: r={r:.3f}, p={p_val:.4f}, n={n}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

# Створення графіків
pairs_to_plot = [
    ('sleep_hours', 'productivity_score'),
    ('steps_per_day', 'weight_kg'),
    ('calories_intake', 'weight_kg'),
    ('caffeine_mg', 'productivity_score')
]

for x_var, y_var in pairs_to_plot:
    pair_result = corr_listwise[
        ((corr_listwise['variable_x'] == x_var) & (corr_listwise['variable_y'] == y_var)) |
        ((corr_listwise['variable_x'] == y_var) & (corr_listwise['variable_y'] == x_var))
    ]
    
    if len(pair_result) > 0:
        row = pair_result.iloc[0]
        output_path = output_dir / f'scatter_{x_var}_vs_{y_var}.png'
        create_scatter_plot(
            df_listwise, x_var, y_var, 
            row['method'], row['coefficient'], row['p_value'], row['n'],
            output_path
        )
        print(f"[OK] {output_path.name}")

# Матриця кореляцій
corr_matrix = df_listwise.corr(method='pearson')

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Коефіцієнт кореляції', rotation=270, labelpad=20)

plt.title('Матриця кореляцій (Pearson, listwise)', pad=20)
plt.tight_layout()

matrix_path = output_dir / 'correlation_matrix.png'
plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] {matrix_path.name}")

matrix_csv_path = output_dir / 'correlation_matrix.csv'
corr_matrix.to_csv(matrix_csv_path)
print(f"[OK] {matrix_csv_path.name}")
print()

# Фінальний чек-лист
print("=" * 80)
print("ЧЕК-ЛИСТ ЗГЕНЕРОВАНИХ ФАЙЛІВ")
print("=" * 80)
print(f"\nПапка: {output_dir.absolute()}\n")

expected_files = [
    'synthetic_data.csv',
    'normality_summary.csv',
    'correlations_listwise.csv',
    'correlation_matrix.csv',
    'correlation_matrix.png',
    'scatter_sleep_hours_vs_productivity_score.png',
    'scatter_steps_per_day_vs_weight_kg.png',
    'scatter_calories_intake_vs_weight_kg.png',
    'scatter_caffeine_mg_vs_productivity_score.png'
]

for i, filename in enumerate(expected_files, 1):
    filepath = output_dir / filename
    status = "[OK]" if filepath.exists() else "[MISS]"
    size = filepath.stat().st_size if filepath.exists() else 0
    size_kb = size / 1024
    print(f"{i:2d}. {status} {filename:50s} ({size_kb:>8.2f} KB)")

total_files = len([f for f in expected_files if (output_dir / f).exists()])
print(f"\n[OK] Всього згенеровано {total_files} файлів")
print("=" * 80)
print("АНАЛІЗ ЗАВЕРШЕНО УСПІШНО!")
print("=" * 80)


#!/usr/bin/env python3
"""Execute notebook and show progress"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

print("="*80)
print("ВИКОНАННЯ НОУТБУКА task1.ipynb")
print("="*80)

# Load notebook
with open('task1.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Execute notebook
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

try:
    print("\nПочинаю виконання комірок...")
    print("(це займе 30-60 секунд)\n")
    
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    
    print("\n" + "="*80)
    print("УСПІХ! Ноутбук виконано без помилок")
    print("="*80)
    
    # Save executed notebook
    with open('task1_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"\nВиконаний ноутбук збережено як: task1_executed.ipynb")
    print(f"Результати у папці: outputs/")
    
except Exception as e:
    print(f"\nПомилка при виконанні: {e}")
    sys.exit(1)


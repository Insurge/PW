#!/usr/bin/env python3
"""Fix notebook encoding issues"""

import json
import sys

# Read notebook
with open('task1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix all cells
changes = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            original = line
            # Replace problematic symbols
            line = line.replace('✓', '[OK]')
            line = line.replace('✗', '[MISS]')
            line = line.replace('×', 'x')
            line = line.replace('\\\\n', '\n')  # Fix escaped newlines
            
            if line != original:
                cell['source'][i] = line
                changes += 1

print(f"Fixed {changes} lines")

# Save fixed notebook
with open('task1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook fixed and saved!")


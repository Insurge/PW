# decision_tree_classifier_report.py
# ----------------------------------
# pip install pandas numpy scikit-learn matplotlib

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

def ensure_dir(p="outputs"):
    Path(p).mkdir(parents=True, exist_ok=True); return Path(p)

def synthetic(seed=42, n=400):
    rng = np.random.default_rng(seed)
    # 2 класи з частковим перекриттям
    x1 = rng.normal(0, 1.0, (n//2, 2))
    x2 = rng.normal(2.0, 1.0, (n//2, 2))
    X = np.vstack([x1, x2])
    y = np.array([0]*(n//2) + [1]*(n//2))
    # додамо 1 категоріальну ознаку (low/med/high) -> one-hot вручну
    cat = rng.choice(["low","med","high"], size=n, p=[0.4,0.4,0.2])
    Xcat = pd.get_dummies(cat, prefix="cat")
    df = pd.DataFrame(X, columns=["feat1","feat2"])
    df = pd.concat([df, Xcat], axis=1)
    df["target"] = y
    return df

def load_or_make(csv, target):
    if csv:
        df = pd.read_csv(csv)
        if target is None:
            target = df.columns[-1]
        return df, target, f"CSV[{target}]"
    else:
        df = synthetic()
        target = "target"
        return df, target, "synthetic_two_class"

def main():
    ap = argparse.ArgumentParser(description="Дерево рішень (класифікація)")
    ap.add_argument("--csv", type=str, default=None, help="Шлях до CSV (останній або --target = ціль)")
    ap.add_argument("--target", type=str, default=None, help="Назва цільового стовпця")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--criterion", type=str, default="gini", choices=["gini","entropy","log_loss"])
    ap.add_argument("--outputs", type=str, default="outputs")
    args = ap.parse_args()

    out = ensure_dir(args.outputs)
    df, target, src = load_or_make(args.csv, args.target)

    # X / y
    y = df[target].values
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=False)  # one-hot на випадок категорій

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    clf = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # текстовий репорт
    rep = classification_report(y_test, y_pred, digits=4)
    (out/"classification_report.txt").write_text(rep, encoding="utf-8")

    # матриця помилок
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix (acc={acc:.3f})")
    plt.savefig(out/"confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # візуалізація дерева
    plt.figure(figsize=(10, 7))
    plot_tree(clf, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True, max_depth=3)
    plt.title("Decision Tree (first 3 levels)")
    plt.savefig(out/"decision_tree.png", bbox_inches="tight")
    plt.close()

    # важливості ознак
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(out/"feature_importances.csv")
    plt.figure()
    imp.head(15).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(out/"feature_importances.png")
    plt.close()

    # Create proper markdown report
    md_lines = [
        "# Дерево рішень — звіт",
        f"**Дані:** {src}",
        f"- criterion={args.criterion}, max_depth={args.max_depth}, test_size={args.test_size}",
        f"- Accuracy (test) = {acc:.4f}",
        "",
        "## Classification report",
        ""
    ]
    
    # Parse classification report into markdown table
    lines = rep.strip().split('\n')
    
    # Find the table part
    table_start = 0
    for i, line in enumerate(lines):
        if 'precision' in line and 'recall' in line and 'f1-score' in line:
            table_start = i
            break
    
    if table_start < len(lines):
        # Header row
        header_line = lines[table_start]
        headers = [h.strip() for h in header_line.split() if h.strip()]
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Data rows - look for all rows with numeric data
        for line in lines[table_start + 1:]:
            if line.strip():
                parts = [p.strip() for p in line.split() if p.strip()]
                if len(parts) >= 4:  # At least 4 columns (class, precision, recall, f1-score, support)
                    # Check if this looks like a data row
                    try:
                        # Try to parse the last few parts as numbers
                        float(parts[-1])  # support should be numeric
                        float(parts[-2])  # f1-score should be numeric
                        float(parts[-3])  # recall should be numeric
                        float(parts[-4])  # precision should be numeric
                        
                        # This is a data row
                        row_values = []
                        for i, part in enumerate(parts):
                            if i == 0:  # class name
                                row_values.append(part)
                            else:  # numeric values
                                try:
                                    val = float(part)
                                    row_values.append(f"{val:.4f}")
                                except ValueError:
                                    row_values.append(part)
                        md_lines.append("| " + " | ".join(row_values) + " |")
                    except (ValueError, IndexError):
                        # Not a data row, skip
                        continue
    
    # Add feature importances table
    md_lines.extend([
        "",
        "## Top Feature Importances",
        ""
    ])
    
    top_features = imp.head(10)
    if not top_features.empty:
        md_lines.append("| Feature | Importance |")
        md_lines.append("| --- | --- |")
        for feature, importance in top_features.items():
            md_lines.append(f"| {feature} | {importance:.4f} |")
    
    (out/"decision_tree_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("Готово ->", out.resolve())
    print(" - classification_report.txt")
    print(" - confusion_matrix.png")
    print(" - decision_tree.png")
    print(" - feature_importances.csv / .png")
    print(" - decision_tree_report.md")

if __name__ == "__main__":
    main()

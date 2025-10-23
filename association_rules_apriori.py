# association_rules_apriori.py
# ----------------------------
# pip install pandas numpy matplotlib
# (без сторонніх бібліотек: Apriori і правила реалізовані вручну)

import argparse
from pathlib import Path
from itertools import combinations, chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p="outputs"):
    Path(p).mkdir(parents=True, exist_ok=True); return Path(p)

def parse_transactions(df, mode):
    """
    Підтримуються 2 формати CSV:
    - mode=itemlist: один стовпець 'items' з рядками типу "milk,bread,eggs"
    - mode=onehot: декілька стовпців 0/1 (назви стовпців = items)
    """
    if mode == "itemlist":
        items = df.iloc[:,0].astype(str).apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
        return [set(x) for x in items.tolist()]
    elif mode == "onehot":
        items = []
        for _, row in df.iterrows():
            items.append(set([c for c, v in row.items() if v==1 or v==True]))
        return items
    else:
        raise ValueError("mode must be 'itemlist' or 'onehot'")

def support_count(transactions, itemset):
    return sum(1 for t in transactions if itemset.issubset(t))

def apriori(transactions, min_support=0.1):
    n = len(transactions)
    # 1. частоти 1-елементних
    item_counts = {}
    for t in transactions:
        for it in t:
            item_counts[frozenset([it])] = item_counts.get(frozenset([it]), 0) + 1
    L = {k for k,v in item_counts.items() if v/n >= min_support}
    levels = [L]
    all_freq = {k: item_counts[k]/n for k in L}

    k = 2
    while True:
        prev = levels[-1]
        # 2. Ck — кандидати шляхом об'єднання L_{k-1}
        candidates = set()
        prev_list = list(prev)
        for i in range(len(prev_list)):
            for j in range(i+1, len(prev_list)):
                union = prev_list[i] | prev_list[j]
                if len(union) == k:
                    # прунинг: усі підмножини повинні бути частими
                    if all(frozenset(s) in prev for s in combinations(union, k-1)):
                        candidates.add(union)

        # 3. підрахунок підтримки
        Lk = set()
        for c in candidates:
            cnt = support_count(transactions, c)
            sup = cnt / n
            if sup >= min_support:
                Lk.add(c)
                all_freq[c] = sup

        if not Lk:
            break
        levels.append(Lk)
        k += 1
    return all_freq

def gen_rules(freq_itemsets, min_conf=0.5, min_lift=1.0):
    """
    Правила: A -> B, де A∩B=∅, A∪B — частий набір.
    Критерії: confidence>=min_conf, lift>=min_lift.
    """
    rows = []
    for itemset, sup_xy in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        # усі не порожні A ⊂ itemset
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = frozenset(A)
                B = itemset - A
                sup_x = freq_itemsets.get(A)
                sup_y = freq_itemsets.get(B)
                if sup_x and sup_y:
                    conf = sup_xy / sup_x
                    lift = conf / sup_y
                    if conf >= min_conf and lift >= min_lift:
                        rows.append({
                            "antecedent": ", ".join(sorted(A)),
                            "consequent": ", ".join(sorted(B)),
                            "support": sup_xy,
                            "confidence": conf,
                            "lift": lift
                        })
    return pd.DataFrame(rows).sort_values(["lift","confidence","support"], ascending=False)

def synthetic_market(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    # Базові ймовірності
    items = ["milk","bread","butter","eggs","coffee","cheese","apples"]
    base_p = dict(milk=0.45,bread=0.55,butter=0.25,eggs=0.35,coffee=0.30,cheese=0.20,apples=0.40)
    tx = []
    for _ in range(n):
        t=set()
        for it in items:
            if rng.random() < base_p[it]:
                t.add(it)
        # ін’єкція правил: milk+bread -> butter ; coffee -> (bread)
        if "milk" in t and "bread" in t and rng.random()<0.45: t.add("butter")
        if "coffee" in t and rng.random()<0.35: t.add("bread")
        tx.append(t)
    return tx

def plot_top_items(outdir, transactions, top=10):
    counts={}
    for t in transactions:
        for it in t: counts[it]=counts.get(it,0)+1
    s=pd.Series(counts).sort_values(ascending=False).head(top)
    plt.figure()
    s.plot(kind="bar")
    plt.ylabel("Count")
    plt.title("Top items")
    plt.tight_layout()
    plt.savefig(outdir/"top_items.png")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Асоціативні правила (Apriori)")
    ap.add_argument("--csv", type=str, default=None, help="Шлях до CSV")
    ap.add_argument("--mode", type=str, default="itemlist", choices=["itemlist","onehot"], help="Формат CSV")
    ap.add_argument("--min_support", type=float, default=0.1)
    ap.add_argument("--min_conf", type=float, default=0.6)
    ap.add_argument("--min_lift", type=float, default=1.0)
    ap.add_argument("--outputs", type=str, default="outputs")
    args = ap.parse_args()

    out = ensure_dir(args.outputs)

    if args.csv:
        df = pd.read_csv(args.csv)
        transactions = parse_transactions(df, args.mode)
        src = f"CSV({args.mode})"
    else:
        transactions = synthetic_market()
        src = "synthetic_market"

    plot_top_items(out, transactions)

    freq = apriori(transactions, min_support=args.min_support)
    freq_df = pd.DataFrame(
        [{"itemset": ", ".join(sorted(list(k))), "support": v} for k,v in freq.items()]
    ).sort_values("support", ascending=False)
    freq_df.to_csv(out/"frequent_itemsets.csv", index=False)

    rules = gen_rules(freq, min_conf=args.min_conf, min_lift=args.min_lift)
    rules.to_csv(out/"association_rules.csv", index=False)

    # Create proper markdown report
    md_lines = [
        "# Асоціативні правила (Apriori)",
        f"**Джерело даних:** {src}",
        f"- min_support={args.min_support}, min_conf={args.min_conf}, min_lift={args.min_lift}",
        "",
        "## Топ частих наборів",
        ""
    ]
    
    # Add frequent itemsets table
    freq_top = freq_df.head(15)
    if not freq_top.empty:
        md_lines.append("| " + " | ".join(freq_top.columns) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(freq_top.columns)) + " |")
        for _, row in freq_top.iterrows():
            row_values = []
            for col in freq_top.columns:
                if col == "support":
                    row_values.append(f"{row[col]:.4f}")
                else:
                    row_values.append(str(row[col]))
            md_lines.append("| " + " | ".join(row_values) + " |")
    
    md_lines.extend([
        "",
        "## Топ правил (за lift)",
        ""
    ])
    
    # Add association rules table
    rules_top = rules.head(20)
    if not rules_top.empty:
        md_lines.append("| " + " | ".join(rules_top.columns) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(rules_top.columns)) + " |")
        for _, row in rules_top.iterrows():
            row_values = []
            for col in rules_top.columns:
                if col in ["support", "confidence", "lift"]:
                    row_values.append(f"{row[col]:.4f}")
                else:
                    row_values.append(str(row[col]))
            md_lines.append("| " + " | ".join(row_values) + " |")
    
    (out/"apriori_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("Готово ->", out.resolve())
    print(" - frequent_itemsets.csv")
    print(" - association_rules.csv")
    print(" - top_items.png")
    print(" - apriori_report.md")

if __name__ == "__main__":
    main()

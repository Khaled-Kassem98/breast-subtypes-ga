# src/compare_step05_vs_step07.py
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
P = {k: ROOT / v for k, v in cfg["paths"].items()}
RES05 = P["results"] / "baseline"
RES07 = P["results"] / "model_selection"
OUT   = P["results"] / "compare"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- load data ----------
# baselines (test split only)
b_metrics = pd.read_csv(RES05 / "baseline_metrics.tsv", sep="\t")
b_test = b_metrics[b_metrics["split"] == "test"].copy()

# step07 summary
summary = json.load(open(RES07 / "summary.json", "r", encoding="utf-8"))
m07_test = next(m for m in summary["metrics"] if m["split"] == "test")
m07_refit = summary.get("test_after_refit", None)

# build comparison table
metrics = ["f1_macro", "balanced_acc", "accuracy", "auc_macro_ovr"]
rows = []

for _, r in b_test.iterrows():
    rows.append({
        "method": f"baseline:{r['model']}",
        **{k: float(r[k]) if k in r and pd.notna(r[k]) else np.nan for k in metrics}
    })

rows.append({"method": "GA+model:test", **{k: m07_test.get(k, np.nan) for k in metrics}})
if m07_refit:
    rows.append({"method": "GA+model:test_refit", **{k: m07_refit.get(k, np.nan) for k in metrics}})

cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(OUT / "compare_table.tsv", sep="\t", index=False)

# ---------- plot 1: metrics bar chart ----------
def plot_metric(metric: str, df: pd.DataFrame, out_png: Path):
    names = df["method"].tolist()
    vals = df[metric].astype(float).tolist()
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, 0.9*len(names)+2), 4))
    ax.bar(x, vals)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}: baseline vs GA")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)

for m in metrics:
    plot_metric(m, cmp_df, OUT / f"compare_{m}.png")

# ---------- pick best baseline by F1 for per-class comparison ----------
best_row = b_test.sort_values("f1_macro", ascending=False).iloc[0]
best_model = best_row["model"]
# load per-class reports
b_rep = json.load(open(RES05 / f"{best_model}_test_report.json", "r", encoding="utf-8"))
# prefer refit comparison if available
if m07_refit and (RES07 / f"{summary['model']}_test_after_refit_report.json").exists():
    m07_rep = json.load(open(RES07 / f"{summary['model']}_test_after_refit_report.json", "r", encoding="utf-8"))
    tag = "test_refit"
else:
    m07_rep = json.load(open(RES07 / f"{summary['model']}_test_report.json", "r", encoding="utf-8"))
    tag = "test"

classes = cfg["classes"]
B = [b_rep.get(c, {}).get("f1-score", np.nan) for c in classes]
G = [m07_rep.get(c, {}).get("f1-score", np.nan) for c in classes]

# ---------- plot 2: per-class F1 (best baseline vs GA) ----------
x = np.arange(len(classes)); w = 0.35
fig, ax = plt.subplots(figsize=(max(6, 0.9*len(classes)+2), 4))
ax.bar(x - w/2, B, width=w, label=f"baseline:{best_model}")
ax.bar(x + w/2, G, width=w, label=f"GA+{summary['model']}:{tag}")
ax.set_xticks(x); ax.set_xticklabels(classes, rotation=0)
ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
ax.set_title("Per-class F1: best baseline vs GA")
ax.legend()
for i, (b, g) in enumerate(zip(B, G)):
    if not np.isnan(b): ax.text(i - w/2, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    if not np.isnan(g): ax.text(i + w/2, g + 0.01, f"{g:.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "compare_per_class_f1.png", dpi=300); plt.close(fig)

print("Wrote:")
print(OUT / "compare_table.tsv")
for m in metrics:
    print(OUT / f"compare_{m}.png")
print(OUT / "compare_per_class_f1.png")

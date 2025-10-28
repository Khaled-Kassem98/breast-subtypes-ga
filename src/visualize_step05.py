# visualize_step05.py
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml","r",encoding="utf-8"))
P = {k: ROOT / v for k, v in cfg["paths"].items()}
RES = P["results"] / "baseline"
RES.mkdir(parents=True, exist_ok=True)
classes = cfg["classes"]

# ---------- 1) Macro-F1 and Balanced Accuracy (grouped bars) ----------
df = pd.read_csv(RES / "baseline_metrics.tsv", sep="\t")
models = list(df["model"].unique())
splits = ["train", "val", "test"]

def grouped_bar(metric: str, out_png: Path):
    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, s in enumerate(splits):
        vals = []
        for m in models:
            row = df[(df.model == m) & (df.split == s)]
            vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + i * w, vals, width=w, label=s)
    ax.set_xticks(x + w); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric); ax.set_ylim(0, 1.0); ax.legend()
    ax.set_title(f"{metric} by split")
    fig.tight_layout(); fig.savefig(out_png, dpi=300)

grouped_bar("f1_macro", RES / "plot_f1_macro.png")
grouped_bar("balanced_acc", RES / "plot_balanced_acc.png")

# ---------- 2) Per-class F1 heatmap on TEST ----------
# collect per-class f1 from each model's test report
M = np.zeros((len(classes), len(models))) * np.nan
for j, m in enumerate(models):
    rp = RES / f"{m}_test_report.json"
    if not rp.exists(): continue
    rep = json.load(open(rp, "r", encoding="utf-8"))
    for i, c in enumerate(classes):
        if c in rep and "f1-score" in rep[c]:
            M[i, j] = rep[c]["f1-score"]

fig, ax = plt.subplots(figsize=(0.9*len(models)+2, 0.6*len(classes)+2))
im = ax.imshow(M, aspect="auto")
ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, ha="right")
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_title("Per-class F1 (test)")
for i in range(len(classes)):
    for j in range(len(models)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(RES / "heatmap_f1_per_class_test.png", dpi=300)

print("Wrote:",
      RES / "plot_f1_macro.png",
      RES / "plot_balanced_acc.png",
      RES / "heatmap_f1_per_class_test.png")

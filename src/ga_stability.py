# src/ga_stability.py
from pathlib import Path
import pandas as pd

proc = Path("data/processed")
masks = sorted(proc.glob("ga_mask_k*.tsv"))
df = [pd.read_csv(p, sep="\t", index_col=0).rename(columns={"selected": p.stem}) for p in masks]
F = pd.concat(df, axis=1).fillna(0)
F["freq"] = F.mean(axis=1)
F.sort_values("freq", ascending=False).to_csv(proc / "ga_selection_frequency.tsv", sep="\t")
print("Wrote", proc / "ga_selection_frequency.tsv")

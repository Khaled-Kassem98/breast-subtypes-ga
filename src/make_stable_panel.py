from pathlib import Path
import pandas as pd
from utils import load_cfg

cfg = load_cfg("config.yaml")
proc = Path(cfg["paths"]["processed"])

df = pd.read_csv(proc / "ga_selection_frequency.tsv", sep="\t", index_col=0)
genes = df.index[df["freq"] >= 0.5].tolist()

out = proc / f"ga_genes_k{len(genes)}.txt"
out.write_text("\n".join(genes), encoding="utf-8")
print("Wrote", out, "N=", len(genes))

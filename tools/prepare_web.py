# tools/prepare_web.py
from pathlib import Path
import sys,json, pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_cfg, read_matrix

cfg = load_cfg("config.yaml")
P = {k:Path(v) for k,v in cfg["paths"].items()}
DOCS = Path("docs"); (DOCS/"data").mkdir(parents=True, exist_ok=True)

# matrix preview: top-variance 200 genes × all GSM
X = read_matrix(str(P["processed"]/ "X_cohort.tsv.gz"))   # genes × gsm
y = pd.read_csv(P["processed"]/ "y.tsv", sep="\t").drop_duplicates("gsm").set_index("gsm")["label"]

var = X.var(axis=1).sort_values(ascending=False)
genes = var.index[:200]
Xv = X.loc[genes]
Xv = ((Xv - Xv.mean(1).values.reshape(-1,1)) / Xv.std(1).replace(0,1).values.reshape(-1,1)).clip(-3,3)

# write compact columnar JSON
(Xv.T.reset_index().rename(columns={"index":"gsm"})
    .to_json(DOCS/"data"/"Xv.json", orient="records"))
pd.Series(genes, name="gene").to_json(DOCS/"data"/"genes.json", orient="values")
y.to_json(DOCS/"data"/"labels.json", orient="index")

# GA + model results (if present)
out = {}
for p in ["results/baseline/compare_step05_vs_step07.tsv",
          "data/processed/ga_selection_frequency.tsv"]:
    q = Path(p)
    if q.exists():
        out[q.name] = pd.read_csv(q, sep="\t").to_dict(orient="list")
(Path(DOCS/"data"/"manifest.json")).write_text(json.dumps(out), encoding="utf-8")
print("Wrote docs/data/*.json")

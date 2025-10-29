# tools/prepare_web.py
from pathlib import Path
import sys,json, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_cfg, read_matrix

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "data"
DOCS.mkdir(parents=True, exist_ok=True)

cfg = load_cfg("config.yaml")
P = cfg["paths"]

# matrix (limit rows to keep file small in browser)
X = read_matrix(Path(P["processed"]) / "X.tsv.gz")
genes = X.index.tolist()
gsms  = X.columns.tolist()

topn = 200  # keep
Xv = X.var(axis=1).sort_values(ascending=False).head(topn).index
Xsub = X.loc[Xv]

with open(DOCS / "matrix.json", "w", encoding="utf-8") as f:
    json.dump({
        "genes": Xsub.index.tolist(),
        "gsms": gsms,
        "X": Xsub.values.tolist()
    }, f)

# labels for supervised view
y = pd.read_csv(Path(P["processed"]) / "y.tsv", sep="\t", index_col=0).iloc[:, 0]
y = y.reindex(gsms).fillna("NA").astype(str)
with open(DOCS / "labels.json", "w", encoding="utf-8") as f:
    json.dump({"labels": y.tolist()}, f)

# latest results (baseline+ga) placeholder if not produced yet
(DOCS / "results.json").write_text("{}", encoding="utf-8")

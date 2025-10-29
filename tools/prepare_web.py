# tools/prepare_web.py
from pathlib import Path
import sys, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_cfg, read_matrix

DOCS = ROOT / "docs" / "data"
DOCS.mkdir(parents=True, exist_ok=True)

cfg = load_cfg("config.yaml")
P = cfg["paths"]

# Pick matrix file
candidates = [
    Path(P["processed"]) / "X.tsv.gz",
    Path(P["processed"]) / "X_cohort.tsv.gz",
]
mtx_path = next((p for p in candidates if p.exists()), None)
if mtx_path is None:
    raise FileNotFoundError("processed/X.tsv.gz or processed/X_cohort.tsv.gz not found")

X = read_matrix(str(mtx_path))
genes = X.index.tolist()
gsms  = X.columns.tolist()

# Keep top-N variable genes
topn = 200
if X.shape[0] > topn:
    var_idx = X.var(axis=1).sort_values(ascending=False).head(topn).index
    Xsub = X.loc[var_idx]
else:
    Xsub = X

(DOCS / "matrix.json").write_text(
    json.dumps({"genes": Xsub.index.tolist(),
                "gsms": gsms,
                "X": Xsub.values.tolist()}),
    encoding="utf-8"
)

# Labels
y_path = Path(P["processed"]) / "y.tsv"
if not y_path.exists():
    raise FileNotFoundError("processed/y.tsv not found")

y = pd.read_csv(y_path, sep="\t", index_col=0).iloc[:, 0]
y = y.reindex(gsms).fillna("NA").astype(str)
(DOCS / "labels.json").write_text(
    json.dumps({"labels": y.tolist()}), encoding="utf-8"
)

# Results (baseline/GA summary) → JSON-safe
summary_path = ROOT / "results" / "model_selection" / "summary.json"
if summary_path.exists():
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    def _np2py(o):
        import numpy as np
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)):  return o.tolist()
        return o

    (DOCS / "results.json").write_text(
        json.dumps(data, indent=2, default=_np2py), encoding="utf-8"
    )
else:
    (DOCS / "results.json").write_text("{}", encoding="utf-8")

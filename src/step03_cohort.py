# src/step03_cohort.py
from pathlib import Path
import pandas as pd

try:
    from .utils import read_matrix, write_matrix
except ImportError:
    from src.utils import read_matrix, write_matrix

def run(cfg, paths):
    P = {k: Path(v) for k, v in paths.items()}
    target = set(cfg["classes"])  # e.g., {"Basal","HER2","LumA","LumB"}

    # load
    X = read_matrix(str(P["processed"] / "X.tsv.gz"))            # genes × GSM
    meta = pd.read_csv(P["processed"] / "metadata.tsv", sep="\t")# columns: gsm,title,label,sample_type,...

    # keep tumor + target labels
    m = meta.copy()
    m = m[m["sample_type"] == "tumor"]
    m = m[m["label"].isin(target)]
    m = m.drop_duplicates(subset=["gsm"])

    # align to expression columns
    cols = [g for g in m["gsm"].tolist() if g in X.columns]
    Xc = X.loc[:, cols]
    y = m.set_index("gsm").loc[cols, "label"].rename("label").reset_index()

    # write
    write_matrix(Xc, str(P["processed"] / "X_cohort.tsv.gz"))
    y.to_csv(P["processed"] / "y.tsv", sep="\t", index=False)

    # diagnostics
    (y["label"].value_counts()
       .to_csv(P["processed"] / "cohort_label_counts.tsv", sep="\t", header=False))
    dropped = pd.DataFrame(
        sorted(set(meta["gsm"]) - set(cols)), columns=["gsm"]
    )
    dropped.to_csv(P["processed"] / "dropped_samples.tsv", sep="\t", index=False)

    return {"n_genes": Xc.shape[0], "n_samples": Xc.shape[1], "labels": dict(y["label"].value_counts())}

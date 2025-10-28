# src/step04_split.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from .utils import read_matrix, write_matrix
except ImportError:
    from src.utils import read_matrix, write_matrix

def run(cfg, paths):
    P = {k: Path(v) for k, v in paths.items()}

    # load genes×samples matrix and labels
    X = read_matrix(str(P["processed"] / "X_cohort.tsv.gz"))
    y = pd.read_csv(P["processed"] / "y.tsv", sep="\t")  # cols: gsm,label

    # align labels to X columns and keep order
    lab = (y.drop_duplicates("gsm")
           .set_index("gsm")["label"]
           .reindex(X.columns))

    # drop columns with missing labels
    mask = lab.notna()
    X = X.loc[:, mask.index[mask]]
    lab = lab[mask]

    # rebuild y with explicit columns
    y = pd.DataFrame({"gsm": lab.index.tolist(), "label": lab.values})

    # ratios
    r = cfg.get("split", {"train": 0.6, "val": 0.2, "test": 0.2})
    train_r, val_r, test_r = r["train"], r["val"], r["test"]
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6

    seed = int(cfg.get("seed", 42))

    # train vs temp (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_r, random_state=seed)
    idx_train, idx_temp = next(sss1.split(np.zeros(len(y)), y["label"]))
    y_train = y.iloc[idx_train]
    y_temp  = y.iloc[idx_temp]

    # split temp into val and test
    temp_r = val_r + test_r
    val_prop = val_r / temp_r
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_prop, random_state=seed + 1)
    idx_val, idx_test = next(sss2.split(np.zeros(len(y_temp)), y_temp["label"]))
    y_val  = y_temp.iloc[idx_val]
    y_test = y_temp.iloc[idx_test]

    # save CSVs
    y_train[["gsm","label"]].to_csv(P["processed"] / "split_train.csv", index=False)
    y_val[["gsm","label"]].to_csv(P["processed"] / "split_val.csv", index=False)
    y_test[["gsm","label"]].to_csv(P["processed"] / "split_test.csv", index=False)

    # optional diagnostics
    (y_train["label"].value_counts()
        .to_csv(P["processed"] / "split_train_counts.tsv", sep="\t", header=False))
    (y_val["label"].value_counts()
        .to_csv(P["processed"] / "split_val_counts.tsv", sep="\t", header=False))
    (y_test["label"].value_counts()
        .to_csv(P["processed"] / "split_test_counts.tsv", sep="\t", header=False))

    return {"n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test)}

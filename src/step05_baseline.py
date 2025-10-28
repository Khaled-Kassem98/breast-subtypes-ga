# src/step05_baseline.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
from joblib import dump

try:
    from .utils import read_matrix, save_fig
except ImportError:
    from src.utils import read_matrix, save_fig

# ---------- helpers ----------

def _load_Xy(P):
    X = read_matrix(str(P["processed"] / "X_cohort.tsv.gz"))      # genes × GSM
    y = pd.read_csv(P["processed"] / "y.tsv", sep="\t")            # gsm,label
    y = y.drop_duplicates("gsm").set_index("gsm")["label"]
    return X, y

def _load_split(P, name):
    df = pd.read_csv(P["processed"] / f"split_{name}.csv")
    return df["gsm"].tolist(), df["label"].tolist()

def _slice(X, y, gsms):
    cols = [g for g in gsms if g in X.columns]
    Xs = X.loc[:, cols].T.values   # samples × genes
    ys = y.reindex(cols).values
    return Xs, ys, cols

def _proba_or_score(clf, X):
    # for AUC: use predict_proba if available, else decision_function
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(X)
        if P.ndim == 1:  # binary (rare here)
            P = np.stack([1 - P, P], axis=1)
        return P
    if hasattr(clf, "decision_function"):
        S = clf.decision_function(X)
        # scale to fake-proba via softmax for multiclass
        if S.ndim == 1:
            S = np.stack([-S, S], axis=1)
        e = np.exp(S - S.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return None

def _eval(clf, X, y_true, labels_order):
    y_pred = clf.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    # AUC macro (OvR)
    proba = _proba_or_score(clf, X)
    try:
        if proba is not None:
            # map labels to indices in labels_order
            y_int = np.array([labels_order.index(c) for c in y_true])
            auc = roc_auc_score(
                y_int,
                proba,
                multi_class="ovr",
                average="macro",
                labels=list(range(len(labels_order)))
            )
            metrics["auc_macro_ovr"] = float(auc)
    except Exception:
        pass
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    report = classification_report(y_true, y_pred, labels=labels_order, zero_division=0, output_dict=True)
    return metrics, cm, report

def _plot_cm(cm, labels, title, out_png):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, str(out_png))
    plt.close(fig)

# ---------- main ----------

def run(cfg, paths):
    P = {k: Path(v) for k, v in paths.items()}
    labels_order = cfg["classes"]  # e.g. ["Basal","HER2","LumA","LumB"]
    k = int(cfg.get("baseline_top_k", 2000))

    # data
    X, y = _load_Xy(P)
    tr_ids, _ = _load_split(P, "train")
    va_ids, _ = _load_split(P, "val")
    te_ids, _ = _load_split(P, "test")

    Xtr, ytr, tr_ids = _slice(X, y, tr_ids)
    Xva, yva, va_ids = _slice(X, y, va_ids)
    Xte, yte, te_ids = _slice(X, y, te_ids)

    n_features = X.shape[0]  # genes
    k_use = min(k, n_features)

    models = {
        "logreg_en": Pipeline([
            ("skb", SelectKBest(score_func=f_classif, k=k_use)),
            ("sc", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga",
                l1_ratio=0.5, C=1.0, max_iter=4000, n_jobs=-1,
                multi_class="multinomial"
            ))
        ]),
        "linsvc": Pipeline([
            ("skb", SelectKBest(score_func=f_classif, k=k_use)),
            ("sc", StandardScaler(with_mean=True)),
            ("clf", LinearSVC(C=1.0))
        ]),
        "rf": Pipeline([
            ("skb", SelectKBest(score_func=f_classif, k=k_use)),
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=cfg["seed"]))
            ]),
    }

    results = []
    outdir = P["results"] / "baseline"
    outdir.mkdir(parents=True, exist_ok=True)

    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)

        for split_name, Xs, ys in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
            metrics, cm, report = _eval(pipe, Xs, ys, labels_order)
            # log
            rec = {"model": name, "split": split_name, **metrics}
            results.append(rec)
            # outputs
            pd.DataFrame(cm, index=labels_order, columns=labels_order)\
              .to_csv(outdir / f"{name}_{split_name}_cm.tsv", sep="\t")
            _plot_cm(cm, labels_order, f"{name} • {split_name}", outdir / f"{name}_{split_name}_cm.png")
            with open(outdir / f"{name}_{split_name}_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        # store predictions on test
        yhat = pipe.predict(Xte)
        pd.DataFrame({"gsm": te_ids, "y_true": yte, "y_pred": yhat})\
          .to_csv(outdir / f"{name}_test_predictions.tsv", sep="\t", index=False)
        # save model
        dump(pipe, outdir / f"{name}.joblib")

    pd.DataFrame(results).to_csv(outdir / "baseline_metrics.tsv", sep="\t", index=False)
    return {"done": True, "models": list(models.keys()), "k_used": k_use}

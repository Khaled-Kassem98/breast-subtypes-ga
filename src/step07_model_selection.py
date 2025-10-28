# src/step07_model_selection.py
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

def _find_ga_genes_file(proc_dir: Path) -> Path:
    # Prefer explicit newest ga_genes_k*.txt
    cands = sorted(proc_dir.glob("ga_genes_k*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError("No ga_genes_k*.txt found in processed/. Run step06 first.")
    return cands[0]

def _load_ga_genes(proc_dir: Path, k_from_cfg: int | None = None) -> list[str]:
    if k_from_cfg:
        p = proc_dir / f"ga_genes_k{k_from_cfg}.txt"
        if p.exists():
            return [g.strip() for g in p.read_text(encoding="utf-8").splitlines() if g.strip()]
    p = _find_ga_genes_file(proc_dir)
    return [g.strip() for g in p.read_text(encoding="utf-8").splitlines() if g.strip()]

def _load_Xy(P):
    X = read_matrix(str(P["processed"] / "X_cohort.tsv.gz"))   # genes × GSM
    y = pd.read_csv(P["processed"] / "y.tsv", sep="\t")        # gsm,label
    y = y.drop_duplicates("gsm").set_index("gsm")["label"]
    return X, y

def _load_split(P, name):
    df = pd.read_csv(P["processed"] / f"split_{name}.csv")
    return df["gsm"].tolist(), df["label"].tolist()

def _slice(X, y, gsms, genes_sel):
    # keep selected genes present in X
    genes = [g for g in genes_sel if g in X.index]
    cols  = [c for c in gsms if c in X.columns]
    Xs = X.loc[genes, cols].T.values   # samples × genes
    ys = y.reindex(cols).values
    return Xs, ys, cols, genes

def _estimator_and_grid(name: str, seed: int):
    name = (name or "logreg_l2").lower()
    if name in ("logreg", "logreg_l2"):
        est = LogisticRegression(
            penalty="l2", solver="lbfgs",
            max_iter=4000, n_jobs=1, random_state=seed
        )
        grid = {"C": [0.1, 0.5, 1.0, 3.0, 10.0]}
        return est, grid
    if name == "logreg_en":
        est = LogisticRegression(
            penalty="l2", solver="lbfgs",
            l1_ratio=0.5, max_iter=4000, n_jobs=1, random_state=seed
        )
        grid = {"C": [0.1, 0.5, 1.0, 3.0], "l1_ratio": [0.1, 0.5, 0.9]}
        return est, grid
    if name == "linsvc":
        est = LinearSVC(random_state=seed)
        grid = {"C": [0.1, 0.5, 1.0, 3.0, 10.0]}
        return est, grid
    if name == "rf":
        est = RandomForestClassifier(n_jobs=1, random_state=seed)
        grid = {"n_estimators": [200, 400, 800],
                "max_depth": [None, 10, 20],
                "max_features": ["sqrt", "log2"]}
        return est, grid
    raise ValueError(f"Unknown model '{name}'")

def _proba_or_score(clf, X):
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(X)
        if P.ndim == 1:
            P = np.stack([1 - P, P], axis=1)
        return P
    if hasattr(clf, "decision_function"):
        S = clf.decision_function(X)
        if S.ndim == 1:
            S = np.stack([-S, S], axis=1)
        e = np.exp(S - S.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return None

def _eval(clf, X, y_true, labels_order):
    y_pred = clf.predict(X)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    proba = _proba_or_score(clf, X)
    try:
        if proba is not None:
            y_int = np.array([labels_order.index(c) for c in y_true])
            out["auc_macro_ovr"] = float(
                roc_auc_score(y_int, proba, multi_class="ovr",
                              average="macro", labels=list(range(len(labels_order))))
            )
    except Exception:
        pass
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    rep = classification_report(y_true, y_pred, labels=labels_order, zero_division=0, output_dict=True)
    return out, cm, rep

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


# ---------- entrypoint ----------

def run(cfg, paths):
    P = {k: Path(v) for k, v in paths.items()}
    labels_order = cfg["classes"]
    seed = int(cfg.get("seed", 42))

    # GA genes
    k_cfg = int(cfg.get("ga", {}).get("k_features", 0)) or None
    genes_sel = _load_ga_genes(P["processed"], k_cfg)

    # data
    X, y = _load_Xy(P)
    tr_ids, _ = _load_split(P, "train")
    va_ids, _ = _load_split(P, "val")
    te_ids, _ = _load_split(P, "test")

    Xtr, ytr, tr_ids, genes_used = _slice(X, y, tr_ids, genes_sel)
    Xva, yva, va_ids, _          = _slice(X, y, va_ids, genes_used)
    Xte, yte, te_ids, _          = _slice(X, y, te_ids, genes_used)

    # model + grid
    model_name = cfg.get("model", "logreg_l2")
    est, grid = _estimator_and_grid(model_name, seed)
    scoring = cfg.get("ga", {}).get("scoring", "f1_macro")
    cv_folds = int(cfg.get("model_cv_folds", cfg.get("ga", {}).get("cv_folds", 5)))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # grid search on train
    gs = GridSearchCV(
        est, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    best_params = gs.best_params_
    best_cv = float(gs.best_score_)

    # evaluate on splits
    outdir = P["results"] / "model_selection"
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for split_name, Xs, ys in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        m, cm, rep = _eval(best, Xs, ys, labels_order)
        m_rec = {"split": split_name, **m}
        metrics.append(m_rec)
        pd.DataFrame(cm, index=labels_order, columns=labels_order)\
          .to_csv(outdir / f"{model_name}_{split_name}_cm.tsv", sep="\t")
        _plot_cm(cm, labels_order, f"{model_name} • {split_name}", outdir / f"{model_name}_{split_name}_cm.png")
        with open(outdir / f"{model_name}_{split_name}_report.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)

    # optional: refit on train+val, eval on test again
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])
    best.fit(Xtrva, ytrva)
    m2, cm2, rep2 = _eval(best, Xte, yte, labels_order)
    pd.DataFrame(cm2, index=labels_order, columns=labels_order)\
      .to_csv(outdir / f"{model_name}_test_after_refit_cm.tsv", sep="\t")
    _plot_cm(cm2, labels_order, f"{model_name} • test (refit tr+val)", outdir / f"{model_name}_test_after_refit_cm.png")
    with open(outdir / f"{model_name}_test_after_refit_report.json", "w", encoding="utf-8") as f:
        json.dump(rep2, f, indent=2)

    # predictions on test
    yhat = best.predict(Xte)
    pd.DataFrame({"gsm": te_ids, "y_true": yte, "y_pred": yhat})\
      .to_csv(outdir / f"{model_name}_test_predictions.tsv", sep="\t", index=False)

    # persist model
    dump(best, outdir / f"{model_name}.joblib")

    # summary
    summary = {
        "model": model_name,
        "best_params": best_params,
        "best_cv_score": best_cv,
        "n_genes_used": len(genes_used),
        "genes_file": _find_ga_genes_file(P["processed"]).name,
        "metrics": metrics,
        "test_after_refit": m2,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

# src/step10_differential.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multitest import multipletests

try:
    from utils import read_matrix, save_fig
except ImportError:
    from src.utils import read_matrix, save_fig

def _bh(p):
    p = np.asarray(p, float)
    p[np.isnan(p)] = 1.0
    _, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    return q

def _volcano(df, title, out_png, topn=12):
    fig, ax = plt.subplots(figsize=(5.2,4.2))
    x = df["log2fc"].values
    y = -np.log10(df["pval"].clip(lower=1e-300)).values
    ax.scatter(x, y, s=8, alpha=0.6)
    sig = (df["fdr"] <= 0.05)
    ax.scatter(x[sig], y[sig], s=10, alpha=0.8)
    ax.axvline( 1.0, ls="--", lw=1); ax.axvline(-1.0, ls="--", lw=1)
    ax.axhline(-np.log10(0.05), ls="--", lw=1)
    ax.set_xlabel("log2FC (class vs rest)")
    ax.set_ylabel("-log10 p")
    ax.set_title(title)
    # annotate top
    lab_df = df.sort_values("pval").head(topn)
    for g, r in lab_df.iterrows():
        ax.text(r["log2fc"], -np.log10(max(r["pval"],1e-300))+0.2, g, fontsize=7)
    fig.tight_layout(); save_fig(fig, str(out_png)); plt.close(fig)

def _prepare_sets(proc_dir: Path, use_train_only: bool):
    X = read_matrix(str(proc_dir / "X_cohort.tsv.gz"))        # genes × GSM
    y = pd.read_csv(proc_dir / "y.tsv", sep="\t").drop_duplicates("gsm").set_index("gsm")["label"]

    if use_train_only and (proc_dir / "split_train.csv").exists():
        tr = pd.read_csv(proc_dir / "split_train.csv")["gsm"].tolist()
        cols = [c for c in X.columns if c in tr]
    else:
        cols = [c for c in X.columns if c in y.index]
    X = X.loc[:, cols]; y = y.loc[cols]
    return X, y

def _t_test_one_vs_rest(X: pd.DataFrame, y: pd.Series, cls: str):
    idx1 = (y == cls).values
    idx0 = ~idx1
    A = X.values[:, idx1]
    B = X.values[:, idx0]
    # log2FC
    m1 = np.nanmean(A, axis=1)
    m0 = np.nanmean(B, axis=1)
    log2fc = m1 - m0
    # Welch t-test
    t, p = ttest_ind(A.T, B.T, equal_var=False, nan_policy="omit")
    # clean constants
    var_all = np.nanvar(X.values, axis=1)
    keep = np.isfinite(p) & (var_all > 0)
    out = pd.DataFrame({
        "gene": X.index,
        "log2fc": log2fc,
        "pval": p
    }).set_index("gene")
    out = out.loc[keep]
    out["fdr"] = _bh(out["pval"].values)
    out.sort_values("fdr", inplace=True)
    return out

def _anova_all_classes(X: pd.DataFrame, y: pd.Series):
    groups = [X.values[:, (y==c).values] for c in pd.unique(y)]
    # scipy f_oneway expects samples along axis=1
    p = []
    for i in range(X.shape[0]):
        try:
            pval = f_oneway(*[g[i, :].ravel() for g in groups]).pvalue
        except Exception:
            pval = np.nan
        p.append(pval)
    df = pd.DataFrame({"gene": X.index, "pval": p}).set_index("gene")
    df["fdr"] = _bh(df["pval"])
    return df.sort_values("fdr")

def run(cfg, paths):
    P = {k: Path(v) for k, v in paths.items()}
    out_dir = P["results"] / "differential"; out_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg["classes"]
    use_train = bool(cfg.get("differential", {}).get("use_train_only", True))

    X, y = _prepare_sets(P["processed"], use_train_only=use_train)

    summary = {}
    for c in classes:
        res = _t_test_one_vs_rest(X, y, c)
        res.to_csv(out_dir / f"de_{c}_vs_rest.tsv", sep="\t")
        _volcano(res, f"{c} vs rest", out_dir / f"volcano_{c}.png")
        summary[c] = {
            "n_sig_fdr_5": int((res["fdr"]<=0.05).sum()),
            "top5_up": res.sort_values(["fdr","log2fc"], ascending=[True,False]).head(5).index.tolist(),
            "top5_down": res.sort_values(["fdr","log2fc"], ascending=[True,True]).head(5).index.tolist()
        }

    # optional ANOVA across all classes
    anova = _anova_all_classes(X, y)
    anova.to_csv(out_dir / "anova_all_classes.tsv", sep="\t")

    return {"use_train_only": use_train, "files": str(out_dir), "summary": summary}

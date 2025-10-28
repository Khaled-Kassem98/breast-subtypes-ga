# src/step09_unsupervised.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    from .utils import read_matrix, save_fig
except ImportError:
    from src.utils import read_matrix, save_fig

def _embed_pca(X, n_components, seed):
    p = PCA(n_components=n_components, random_state=seed)
    Z = p.fit_transform(X)
    return Z, p.explained_variance_ratio_

def _embed_umap(X, seed):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed)
    Z = reducer.fit_transform(X)
    return Z

def _scatter(Z, labels, title, out_png):
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = pd.unique(labels)
    for c in classes:
        idx = (labels == c)
        ax.scatter(Z[idx, 0], Z[idx, 1], s=20, label=str(c), alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("dim1"); ax.set_ylabel("dim2")
    ax.legend(markerscale=1, fontsize=8, loc="best", frameon=False)
    fig.tight_layout()
    save_fig(fig, str(out_png))
    plt.close(fig)

def _cluster_and_score(X, y, k, seed, algo):
    if algo == "kmeans":
        est = KMeans(n_clusters=k, n_init=20, random_state=seed)
    elif algo == "agglo":
        est = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        raise ValueError("algo must be 'kmeans' or 'agglo'")
    lab = est.fit_predict(X)
    # metrics (handle degenerate)
    sil = float(silhouette_score(X, lab)) if k > 1 and len(np.unique(lab)) > 1 else np.nan
    ari = float(adjusted_rand_score(y, lab))
    nmi = float(normalized_mutual_info_score(y, lab))
    return lab, {"algo": algo, "k": k, "silhouette": sil, "ARI": ari, "NMI": nmi}

def run(cfg, paths):
    P = {k: Path(v) for k, v in paths.items()}
    seed = int(cfg.get("seed", 42))
    classes = cfg["classes"]

    # ---- data ----
    Xg = read_matrix(str(P["processed"] / "X_cohort.tsv.gz"))  # genes × GSM
    y_tbl = pd.read_csv(P["processed"] / "y.tsv", sep="\t").drop_duplicates("gsm").set_index("gsm")["label"]
    # align
    gsm = [c for c in Xg.columns if c in y_tbl.index]
    X = Xg.loc[:, gsm].T.values
    y = y_tbl.loc[gsm].values

    # ---- scale + PCA50 ----
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    Z50, var = _embed_pca(Xs, n_components=min(50, Xs.shape[1], Xs.shape[0]-1), seed=seed)

    out = P["results"] / "unsupervised"
    out.mkdir(parents=True, exist_ok=True)

    # ---- 2D embeddings ----
    Zp2, var2 = _embed_pca(Xs, n_components=2, seed=seed)
    pd.DataFrame(Zp2, index=gsm, columns=["pc1","pc2"]).to_csv(out / "embed_pca2.csv", index=True)
    _scatter(Zp2, y, f"PCA2 • var={var2.sum():.2f}", out / "embed_pca2.png")

    if HAVE_UMAP:
        Zu2 = _embed_umap(Z50, seed)
        pd.DataFrame(Zu2, index=gsm, columns=["umap1","umap2"]).to_csv(out / "embed_umap2.csv", index=True)
        _scatter(Zu2, y, "UMAP2 (PCA50→UMAP2)", out / "embed_umap2.png")

    # ---- clustering at k = #classes ----
    k_target = len(classes)
    rows = []
    for algo in ["kmeans", "agglo"]:
        lab, m = _cluster_and_score(Z50, y, k_target, seed, algo)
        rows.append(m)
        pd.DataFrame({"gsm": gsm, "cluster": lab}).to_csv(out / f"clusters_{algo}_k{k_target}.csv", index=False)

    # ---- sweep k=2..8 for both algos ----
    for algo in ["kmeans", "agglo"]:
        for k in range(2, 9):
            lab, m = _cluster_and_score(Z50, y, k, seed, algo)
            rows.append(m)

    # ---- metrics table ----
    dfm = pd.DataFrame(rows)
    dfm.to_csv(out / "metrics.tsv", sep="\t", index=False)

    # ---- silhouette vs k plot ----
    for metric in ["silhouette", "ARI", "NMI"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for algo in ["kmeans", "agglo"]:
            d = dfm[dfm["algo"] == algo].copy()
            ax.plot(d["k"], d[metric], marker="o", label=algo)
        ax.set_xlabel("k"); ax.set_ylabel(metric); ax.set_title(f"{metric} vs k")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, str(out / f"{metric}_vs_k.png"))
        plt.close(fig)

    # brief return
    best = (dfm.sort_values(["ARI","NMI","silhouette"], ascending=False)
               .head(1).to_dict(orient="records")[0])
    return {"pca2_var": float(var2.sum()), "best": best, "files": str(out)}

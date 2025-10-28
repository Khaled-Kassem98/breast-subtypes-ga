# src/step08_bio_checks.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from gseapy import enrichr
    HAVE_GSEAPY = True
except Exception:
    HAVE_GSEAPY = False

try:
    from .utils import read_matrix, save_fig
except ImportError:
    from src.utils import read_matrix, save_fig

# PAM50 panel
PAM50 = [
    "BAG1","BCL2","BIRC5","BLVRA","CCNB1","CDC20","CDC6","CDH3","CENPF","CEP55",
    "EGFR","ERBB2","ESR1","EXO1","FGFR4","FOXA1","GPR160","GRB7","KRT14","KRT17",
    "KRT5","KRT8","KRT18","KRT19","MAPT","MELK","MIA","MKI67","MLPH","MMP11",
    "MYBL2","NAT1","NDC80","NUF2","ORC6","PGR","PHGDH","PTTG1","RRM2","SLC39A6",
    "TMEM45B","TYMS","UBE2C","UBE2T","XBP1","KIF2C","KIF14","BUB1B","CXXC5","ACTR3B"
]

# simple subtype markers
MARKERS = {
    "Basal": ["KRT5","KRT14","KRT17","EGFR"],
    "HER2":  ["ERBB2","GRB7","FGFR4"],
    "LumA":  ["ESR1","PGR","FOXA1","BCL2","GATA3","KRT8","KRT18"],
    "LumB":  ["MKI67","CCNB1","AURKA","BIRC5","ESR1","FOXA1"]
}

def _load_ga_genes(proc_dir: Path, prefer_k: int | None = None) -> list[str]:
    if prefer_k:
        p = proc_dir / f"ga_genes_k{prefer_k}.txt"
        if p.exists():
            return [g.strip() for g in p.read_text(encoding="utf-8").splitlines() if g.strip()]
    cands = sorted(proc_dir.glob("ga_genes_k*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError("ga_genes_k*.txt not found. Run step06.")
    return [g.strip() for g in cands[0].read_text(encoding="utf-8").splitlines() if g.strip()]

def _plot_marker_heatmap(X, y_map, genes, classes, out_png: Path):
    g = [gg for gg in genes if gg in X.index]
    if not g:
        return
    # per-class means
    M = []
    for c in classes:
        cols = [s for s, lab in y_map.items() if lab == c and s in X.columns]
        if len(cols) == 0:
            M.append(np.full(len(g), np.nan))
        else:
            M.append(X.loc[g, cols].mean(axis=1).values)
    M = np.stack(M, axis=1)  # genes × classes

    fig, ax = plt.subplots(figsize=(max(6, 0.35*len(classes)+2), max(5, 0.22*len(g)+2)))
    im = ax.imshow(M, aspect="auto")
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=0)
    ax.set_title("Marker genes: mean z-score by subtype")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, str(out_png))
    plt.close(fig)

def _run_enrichment(genes: list[str], out_dir: Path) -> dict:
    libs = ["MSigDB_Hallmark_2020","KEGG_2019_Human","GO_Biological_Process_2021"]
    out = {}
    if not HAVE_GSEAPY:
        (out_dir / "enrichment_skipped.txt").write_text("gseapy not available", encoding="utf-8")
        return out
    for lib in libs:
        try:
            enr = enrichr(gene_list=genes, gene_sets=[lib], organism="Human",
                          outdir=str(out_dir / f"enrich_{lib}"), cutoff=0.05, no_plot=True)
            res = enr.results if hasattr(enr, "results") else getattr(enr, "res2d", pd.DataFrame())
            if res is None or len(res) == 0:
                continue
            # standardize column names
            df = res.rename(columns={
                "Term":"term","Adjusted P-value":"adj_p","P-value":"pvalue",
                "Odds Ratio":"odds","Combined Score":"combined_score","Overlap":"overlap","Genes":"genes"
            })
            df.sort_values(["adj_p","pvalue"], inplace=True)
            df.to_csv(out_dir / f"top_{lib}.tsv", sep="\t", index=False)
            out[lib] = df.head(20).to_dict(orient="list")
        except Exception as e:
            (out_dir / f"enrich_{lib}_error.txt").write_text(str(e), encoding="utf-8")
    return out

def run(cfg, paths):
    P = {k: Path(v) for k, v in paths.items()}
    classes = cfg["classes"]
    proc = P["processed"]; res = P["results"] / "bio"
    res.mkdir(parents=True, exist_ok=True)

    # load data
    X = read_matrix(str(proc / "X_cohort.tsv.gz"))   # genes × GSM (z-scored by gene)
    y = pd.read_csv(proc / "y.tsv", sep="\t")
    y = y.drop_duplicates("gsm").set_index("gsm")["label"]
    y_map = y.to_dict()

    # GA genes
    k_cfg = int(cfg.get("ga", {}).get("k_features", 0)) or None
    ga_genes = _load_ga_genes(proc, k_cfg)

    # overlaps
    pam50_hit = sorted(set(ga_genes) & set(PAM50))
    pd.Series(pam50_hit, name="gene").to_csv(res / "ga_in_PAM50.txt", index=False)

    # per-subtype marker overlap table
    rows = []
    for c, glist in MARKERS.items():
        glist = list(set(glist))
        present = sorted(set(glist) & set(X.index))
        in_ga   = sorted(set(glist) & set(ga_genes))
        rows.append({
            "subtype": c,
            "n_markers_defined": len(glist),
            "n_markers_in_data": len(present),
            "n_markers_in_GA": len(in_ga),
            "markers_in_GA": ",".join(in_ga)
        })
    pd.DataFrame(rows).to_csv(res / "marker_overlap.tsv", sep="\t", index=False)

    # heatmaps per group of markers
    for c, glist in MARKERS.items():
        g = [gg for gg in glist if gg in X.index]
        if g:
            _plot_marker_heatmap(X, y_map, g, classes, res / f"heatmap_markers_{c}.png")

    # combined heatmap for all unique markers
    all_markers = sorted(set(sum(MARKERS.values(), [])))
    all_markers = [g for g in all_markers if g in X.index]
    if all_markers:
        _plot_marker_heatmap(X, y_map, all_markers, classes, res / "heatmap_markers_all.png")

    # enrichment on GA genes
    enr = _run_enrichment(ga_genes, res)

    # small bar for PAM50 overlap
    try:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["PAM50 overlap"], [len(pam50_hit)])
        ax.set_ylim(0, max(1, len(pam50_hit)+1))
        for i, v in enumerate([len(pam50_hit)]):
            ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=9)
        ax.set_title("GA genes ∩ PAM50")
        save_fig(fig, str(res / "pam50_overlap_bar.png"))
        plt.close(fig)
    except Exception:
        pass

    # summary json
    summary = {
        "n_ga_genes": len(ga_genes),
        "pam50_overlap": len(pam50_hit),
        "markers_in_ga": {k: len(set(v) & set(ga_genes)) for k, v in MARKERS.items()},
        "enrichment_sets": list(enr.keys()) if enr else [],
    }
    (res / "bio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

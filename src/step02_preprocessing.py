# src/step02_preprocessing.py
from pathlib import Path
import pandas as pd
import numpy as np
import GEOparse
import gzip,io
import csv

try:
    from .utils import read_matrix, write_matrix, is_log2_scaled, zscore_rows
except ImportError:
    from src.utils import read_matrix, write_matrix, is_log2_scaled, zscore_rows

def _find_gpl_file(raw_dir: Path) -> Path:
    # prefer local; else download once
    cands = [
        raw_dir / "GPL570_family.soft.gz",
        raw_dir / "GPL570_family.soft",
        raw_dir / "GPL570.annot.gz",
        raw_dir / "GPL570.annot",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("Put GPL570_family.soft.gz or GPL570.annot(.gz) in data/raw")

def _probe_to_gene_map(ann_path: Path) -> pd.Series:
    # Support SOFT or plain table
    # Case 1: SOFT
    if ann_path.name.endswith((".soft", ".soft.gz")):
        gpl = GEOparse.get_GEO(filepath=str(ann_path))
        df = gpl.table.copy()
        lc = {c.lower(): c for c in df.columns}
        idc = lc.get("id") or lc.get("id_ref") or "ID"
        symc = lc.get("gene symbol") or lc.get("genesymbol") or lc.get("gene_symbol")
        if symc is None:
            symc = next((c for c in df.columns if "gene" in c.lower() and "symbol" in c.lower()), None)
        if symc is None:
            raise KeyError("Gene Symbol column not found in GPL SOFT.")
        df = df[[idc, symc]].rename(columns={idc: "probe", symc: "symbol"})
        return _clean_map(df)

    # Case 2: .annot(.gz) with preamble → find header, then parse
    text = _read_text_preserve(ann_path)
    lines = text.splitlines()
    header_i, header_cols = _find_header(lines)
    if header_i is None:
        raise RuntimeError("Could not locate header line with probe ID and Gene Symbol in GPL570.annot.gz")
    body = "\n".join(lines[header_i + 1:])
    df = pd.read_csv(
        io.StringIO(body),
        sep="\t",
        header=None,
        names=header_cols,
        dtype=str,
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    # normalize column names
    lc = {c.lower(): c for c in df.columns}
    idc = lc.get("id") or lc.get("probe id") or lc.get("probe_id") or "id_ref" or "id"
    symc = lc.get("gene symbol") or lc.get("symbol") or lc.get("gene_symbol")
    if symc is None:
        symc = next((c for c in df.columns if "gene" in c.lower() and "symbol" in c.lower()), None)
    if idc not in df.columns or symc is None:
        raise KeyError("Probe ID or Gene Symbol column not found after parsing GPL570.annot.gz")

    df = df[[idc, symc]].rename(columns={idc: "probe", symc: "symbol"})
    return _clean_map(df)

def _read_text_preserve(path: Path) -> str:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        return f.read()
def _find_header(lines: list[str]):
    # pick the first tabbed line containing both ID and Gene Symbol (case-insensitive)
    for i, ln in enumerate(lines):
        if "\t" not in ln:
            continue
        low = ln.lower()
        if "gene" in low and "symbol" in low and ("id\t" in low or low.startswith("id")):
            cols = [c.strip() for c in ln.rstrip("\n").split("\t")]
            return i, cols
    return None, None
def _clean_map(df: pd.DataFrame) -> pd.Series:
    df["symbol"] = (
        df["symbol"].astype(str)
        .str.split(" /// ").str[0]
        .str.split(" // ").str[0]
        .str.strip()
    )
    df.replace({"symbol": {"", "NA", "na", "NaN"}}, np.nan, inplace=True)
    df.dropna(subset=["symbol"], inplace=True)
    df.drop_duplicates(subset=["probe"], inplace=True)
    return df.set_index("probe")["symbol"]
def run(cfg, paths):
    raw_dir = Path(paths["raw"])
    X = read_matrix(str(Path(paths["interim"]) / "X_raw.tsv.gz"))

    # log2 check
    if not is_log2_scaled(X):
        # series matrix should already be log2; if not, log2(x+1)
        X = np.log2(X + 1.0)
        X = pd.DataFrame(X, index=X.index, columns=X.columns)

    # map probes -> genes
    ann_path = _find_gpl_file(Path(paths["raw"]))
    p2g = _probe_to_gene_map(ann_path)
    common = X.index.intersection(p2g.index)
    Xp = X.loc[common].copy()
    Xp["gene"] = p2g.loc[common].values

    # collapse to gene by median
    Xg = Xp.groupby("gene").median(numeric_only=True)

    # z-score per gene
    #Xg = zscore_rows(Xg)

    write_matrix(Xg, str(Path(paths["processed"]) / "X.tsv.gz"))
    return {"n_genes": Xg.shape[0], "n_samples": Xg.shape[1]}

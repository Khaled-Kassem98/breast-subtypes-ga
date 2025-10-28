# src/step01_reading_data.py
from __future__ import annotations

import gzip, io, re
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import GEOparse

try:
    from .utils import write_matrix
except ImportError:
    from src.utils import write_matrix


# ---------- paths ----------

def _series_matrix_path(raw_dir: Path, gse_id: str) -> Path:
    candidates = [
        raw_dir / f"{gse_id}_series_matrix.txt.gz",
        raw_dir / f"{gse_id}_series_matrix.txt",
        raw_dir / "raw" / f"{gse_id}_series_matrix.txt.gz",
        raw_dir / "raw" / f"{gse_id}_series_matrix.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Place {gse_id}_series_matrix.txt[.gz] under "
                            f"{raw_dir} (or {raw_dir / 'raw'}).")

def _family_soft_path(raw_dir: Path, gse_id: str) -> Path:
    candidates = [
        raw_dir / f"{gse_id}_family.soft.gz",
        raw_dir / f"{gse_id}_family.soft",
        raw_dir / "raw" / f"{gse_id}_family.soft.gz",
        raw_dir / "raw" / f"{gse_id}_family.soft",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Place {gse_id}_family.soft[.gz] under {raw_dir} (or {raw_dir / 'raw'}).")


# ---------- loaders ----------

def _read_series_matrix(gz_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    b = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    e = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    tbl = "".join(lines[b + 1 : e])
    df = pd.read_csv(io.StringIO(tbl), sep="\t").set_index("ID_REF")
    gsms = [c for c in df.columns if c.startswith("GSM")]
    return df[gsms], lines


def _parse_metadata_from_family_soft(soft_path: Path, gsms_order: list[str]) -> pd.DataFrame:
    # parse local SOFT without downloading
    gse = GEOparse.get_GEO(filepath=str(soft_path))

    rows = []
    for gsm_id in gsms_order:
        if gsm_id not in gse.gsms:
            rows.append({"gsm": gsm_id, "title": "", "label": None,
                         "sample_type": "tumor", "raw_text": ""})
            continue

        gsm = gse.gsms[gsm_id]
        title = gsm.metadata.get("title", [""])[0]
        source = gsm.metadata.get("source_name_ch1", [""])[0]
        char_keys = [k for k in gsm.metadata if k.startswith("characteristics_ch1")]
        chvals = []
        for k in char_keys:
            chvals.extend(gsm.metadata.get(k, []))

        parts = [title, source] + chvals
        blob = " | ".join([s for s in parts if s]).lower()

        s_type = "tumor"
        if re.search(r"\bnormal\b|adjacent normal|healthy|control", blob): s_type = "normal"
        if "cell line" in blob or "cell-line" in blob: s_type = "cell_line"

        y = None
        if "luminal a" in blob or "luma" in blob: y = "LumA"
        elif "luminal b" in blob or "lumb" in blob: y = "LumB"
        elif "her2" in blob or "erbb2" in blob: y = "HER2"
        elif "basal" in blob or "triple-negative" in blob or re.search(r"\btn\b", blob): y = "Basal"

        rows.append({
            "gsm": gsm_id,
            "title": title,
            "label": y,
            "sample_type": s_type,
            "raw_text": " | ".join([s for s in parts if s]),
        })
    return pd.DataFrame(rows)


# ---------- entrypoint ----------

def run(cfg: dict, paths: dict) -> dict:
    raw_dir = Path(paths["raw"])
    gse_id = cfg["gse"]
    print("RAW DIR:", raw_dir)
    # expression matrix from Series Matrix

    smx = _series_matrix_path(raw_dir, gse_id)
    print("SERIES MATRIX:", smx)
    X, _ = _read_series_matrix(smx)
    gsms = list(X.columns)
    write_matrix(X, str(Path(paths["interim"]) / "X_raw.tsv.gz"))

    # metadata from local SOFT family file
    soft = _family_soft_path(raw_dir, gse_id)
    print("FAMILY SOFT:", soft)
    meta = _parse_metadata_from_family_soft(soft, gsms)
    meta.to_csv(Path(paths["processed"]) / "metadata.tsv", sep="\t", index=False)

    # column order for downstream alignment
    pd.Series(gsms, name="gsm").to_csv(Path(paths["processed"]) / "gsm_order.tsv",
                                       sep="\t", index=False)
    from collections import Counter
    out_dir = Path(paths["processed"])

    # counts
    meta["label"].fillna("NA").value_counts().to_csv(out_dir / "label_counts.tsv", sep="\t", header=False)
    meta["sample_type"].value_counts().to_csv(out_dir / "sample_type_counts.tsv", sep="\t", header=False)

    # crude replicate detection by number in title, e.g. "Sample97"
    import re
    def subj_id(t):
        if not isinstance(t, str): return None
        m = re.search(r"sample\s*(\d+)", t, flags=re.I)
        return m.group(1) if m else None

    meta["subject_id"] = meta["title"].apply(subj_id)
    dup = meta[meta["subject_id"].notna()].groupby("subject_id")["gsm"].apply(list).reset_index()
    dup = dup[dup["gsm"].apply(len) > 1]
    dup.to_csv(out_dir / "replicate_groups.tsv", sep="\t", index=False)
    return {"n_probes": X.shape[0], "n_samples": X.shape[1]}

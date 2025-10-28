# src/utils.py
from __future__ import annotations
import os, json, random, logging, time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd
from contextlib import contextmanager

# project root = repo root (utils.py is in src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_cfg(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML and resolve paths."""
    p = (PROJECT_ROOT / path) if not Path(path).is_absolute() else Path(path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = resolve_paths(cfg.get("paths", {}))
    return cfg

def resolve_paths(paths: Dict[str, str]) -> Dict[str, str]:
    """Make absolute dirs from cfg and create them."""
    out = {}
    for k, v in paths.items():
        pv = PROJECT_ROOT / v if not Path(v).is_absolute() else Path(v)
        pv.mkdir(parents=True, exist_ok=True)
        out[k] = str(pv)
    return out

def set_seed(seed: int) -> None:
    """Full reproducibility best-effort."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def setup_logging(log_path: Optional[str] = None, level: int = logging.INFO) -> None:
    """Console logging, optional file."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    if log_path:
        fp = PROJECT_ROOT / log_path if not Path(log_path).is_absolute() else Path(log_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)

def read_matrix(path: str) -> pd.DataFrame:
    """TSV/TSV.GZ with index in col0."""
    p = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    return pd.read_csv(p, sep="\t", index_col=0, compression="infer")

def write_matrix(df: pd.DataFrame, path: str) -> None:
    p = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    comp = "gzip" if str(p).endswith(".gz") else None
    df.to_csv(p, sep="\t", index=True, compression=comp)

def read_json(path: str) -> Any:
    p = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: str, indent: int = 2) -> None:
    p = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)

def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gene standardization."""
    v = df.values
    mu = np.nanmean(v, axis=1, keepdims=True)
    sd = np.nanstd(v, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return pd.DataFrame((v - mu) / sd, index=df.index, columns=df.columns)

def is_log2_scaled(df: pd.DataFrame) -> bool:
    """Heuristic check for log2 microarray scale."""
    a = df.values.ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return False
    q01, q99 = np.quantile(a, [0.01, 0.99])
    return (q99 <= 20) and ((q99 - q01) <= 20)

def save_fig(fig, path: str, dpi: int = 300) -> None:
    p = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")

def ensure_exists(path: str | Path, msg: str = "") -> Path:
    p = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        raise FileNotFoundError(msg or f"Missing: {p}")
    return p

def find_first(root: str | Path, pattern: str) -> Optional[Path]:
    r = PROJECT_ROOT / root if not Path(root).is_absolute() else Path(root)
    for p in r.rglob(pattern):
        if p.is_file():
            return p
    return None
@contextmanager
def timer(name: str):
    """Context timer."""
    t0 = time.time()
    yield
    dt = time.time() - t0
    logging.getLogger(__name__).info(f"{name} done in {dt:.2f}s")


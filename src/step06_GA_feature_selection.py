# src/step06_GA_feature_selection.py
from pathlib import Path
import json, random,warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from deap import base, creator, tools, algorithms

try:
    from .utils import read_matrix, write_matrix
except ImportError:
    from src.utils import read_matrix, write_matrix
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)



# known breast-cancer markers to reward
KNOWN = {
    "BAG1","BCL2","BIRC5","BLVRA","CCNB1","CDC20","CDC6","CDH3","CENPF","CEP55",
    "EGFR","ERBB2","ESR1","EXO1","FGFR4","FOXA1","GRB7","KRT14","KRT17","KRT5",
    "KRT8","KRT18","KRT19","MAPT","MELK","MIA","MKI67","MLPH","MMP11","MYBL2",
    "NAT1","NDC80","NUF2","ORC6","PGR","PHGDH","PTTG1","RRM2","SLC39A6","TMEM45B",
    "TYMS","UBE2C","UBE2T","XBP1","KIF2C","KIF14","BUB1B","CXXC5","ACTR3B",
    "GATA3","AURKA"
}
ALPHA = 0.005  # small bonus weight
# ---------- estimator factory ----------

def _estimator(name: str, seed: int):
    name = (name or "logreg_en").lower()
    if name == "logreg_en":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            max_iter=4000, n_jobs=-1, multi_class="multinomial", random_state=seed
        )
    if name == "linsvc":
        return LinearSVC(C=1.0, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)
    raise ValueError(f"Unknown estimator '{name}'")

# ---------- GA core ----------

def run(cfg, paths,k=300):
    P = {k: Path(v) for k, v in paths.items()}
    ga_cfg = cfg.get("ga", {})
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed)

    # data
    X_full = read_matrix(str(P["processed"] / "X_cohort.tsv.gz"))   # genes × GSM
    y_tbl  = pd.read_csv(P["processed"] / "y.tsv", sep="\t")        # gsm,label

    # splits
    tr_ids = pd.read_csv(P["processed"] / "split_train.csv")["gsm"].tolist()
    # GA evaluates by CV on train only
    cols_tr = [g for g in tr_ids if g in X_full.columns]
    X_tr = X_full.loc[:, cols_tr].T.values        # samples × genes
    y_tr = (y_tbl.drop_duplicates("gsm")
                 .set_index("gsm").loc[cols_tr, "label"].values)

    # features
    genes = X_full.index.tolist()
    n_features = len(genes)
    k = int(ga_cfg.get("k_features", min(300, max(50, n_features // 20))))  # default ~5%
    k = max(5, min(k, n_features))

    est_name = ga_cfg.get("estimator", "logreg_en")
    est = _estimator(est_name, seed)
    scoring = ga_cfg.get("scoring", "f1_macro")
    cv_folds = int(ga_cfg.get("cv_folds", 5))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    pop_size = int(ga_cfg.get("population_size", 80))
    ngen = int(ga_cfg.get("generations", 40))
    cxpb = float(ga_cfg.get("crossover_prob", 0.8))
    mutpb = float(ga_cfg.get("mutation_prob", 0.05))

    # cache evaluations
    score_cache = {}

    def eval_indices(idxs_tuple):
        if idxs_tuple in score_cache:
            return score_cache[idxs_tuple]
        idxs = np.fromiter(idxs_tuple, dtype=int)
        Xs = X_tr[:, idxs]
        try:
            s = cross_val_score(est, Xs, y_tr, cv=cv, scoring=scoring, n_jobs=-1).mean()
        except Exception:
            s = 0.0
        score_cache[idxs_tuple] = s
        return s

    # DEAP setup: individuals are size-k lists of unique indices [0..n_features-1]
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_individual():
        return creator.Individual(sorted(np.random.choice(n_features, size=k, replace=False).tolist()))

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        base = eval_indices(tuple(ind))  # CV f1_macro
        bonus = ALPHA * sum(1 for i in ind if genes[i] in KNOWN)
        return (base + bonus,)

    def cx_set(ind1, ind2):
        # union then sample k unique
        s = set(ind1) | set(ind2)
        if len(s) < k:
            # fill from remaining pool
            pool = list(set(range(n_features)) - s)
            need = k - len(s)
            s |= set(np.random.choice(pool, size=need, replace=False).tolist())
        child = sorted(np.random.choice(list(s), size=k, replace=False).tolist())
        ind1[:] = child
        # symmetric for ind2
        s2 = set(ind1) | set(ind2)
        if len(s2) < k:
            pool = list(set(range(n_features)) - s2)
            need = k - len(s2)
            s2 |= set(np.random.choice(pool, size=need, replace=False).tolist())
        ind2[:] = sorted(np.random.choice(list(s2), size=k, replace=False).tolist())
        return ind1, ind2

    def mut_swap(ind, pm=0.1):
        # replace ~pm*k genes with new unique indices
        m = max(1, int(pm * k))
        current = set(ind)
        pool = list(set(range(n_features)) - current)
        if len(pool) == 0:
            return (ind,)
        repl = np.random.choice(range(k), size=m, replace=False)
        new_vals = np.random.choice(pool, size=m, replace=False)
        for pos, nv in zip(repl, new_vals):
            ind[pos] = int(nv)
        ind[:] = sorted(set(ind))  # ensure uniqueness, keep size with fill if needed
        while len(ind) < k:
            cand_pool = list(set(range(n_features)) - set(ind))
            ind.append(int(np.random.choice(cand_pool)))
        ind[:] = sorted(ind[:k])
        return (ind,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_set)
    toolbox.register("mutate", mut_swap, pm=0.10)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # evolve
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=ngen, stats=stats, halloffame=hof, verbose=False)

    best = list(hof[0])
    best_score = float(hof[0].fitness.values[0])
    best_genes = [genes[i] for i in best]

    # outputs
    out_proc = P["processed"]
    out_res = P["results"] / "ga"
    out_proc.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)

    # save selected list
    (out_proc / f"ga_genes_k{k}.txt").write_text("\n".join(best_genes), encoding="utf-8")

    # save mask table
    mask = pd.Series(0, index=genes)
    mask.loc[best_genes] = 1
    mask.rename("selected").to_frame().to_csv(out_proc / f"ga_mask_k{k}.tsv", sep="\t")

    # save reduced matrix for convenience
    X_ga = X_full.loc[best_genes]
    write_matrix(X_ga, str(out_proc / "X_ga.tsv.gz"))

    # logbook
    log_df = pd.DataFrame(log)
    log_df.to_csv(out_res / "ga_log.tsv", sep="\t", index=False)

    meta = {
        "k": k, "best_score": best_score, "estimator": est_name,
        "cv_folds": cv_folds, "scoring": scoring,
        "population_size": pop_size, "generations": ngen,
    }
    (out_res / "ga_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"k": k, "best_score": best_score, "genes": len(best_genes)}


# src/check_splits.py
import pandas as pd, pathlib as p
P = p.Path("data/processed")
tr=set(pd.read_csv(P/"split_train.csv")["gsm"])
va=set(pd.read_csv(P/"split_val.csv")["gsm"])
te=set(pd.read_csv(P/"split_test.csv")["gsm"])
print("overlaps:", len(tr&va), len(tr&te), len(va&te))  # must be 0 0 0

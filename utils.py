# utils.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def binary_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def recall_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.001) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    if target_fpr <= fpr.max():
        return float(np.interp(target_fpr, fpr, tpr))
    return 0.0

def prepare_creditcard(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    if "Class" not in df.columns:
        raise ValueError("CreditCard dataset must contain 'Class'.")
    y = df["Class"].astype(int).values
    X_df = df.drop(columns=[c for c in ["Time", "Class"] if c in df.columns]).copy()
    if "Amount" in X_df.columns:
        X_df["Amount"] = StandardScaler().fit_transform(X_df[["Amount"]]).ravel()
    return X_df.values, y, list(X_df.columns), "creditcard"

def prepare_paysim(df: pd.DataFrame, subsample_if_huge: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    need = {"step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud","isFlaggedFraud"}
    if not need.issubset(df.columns): raise ValueError("PaySim dataset missing required columns.")
    if subsample_if_huge and len(df) > 1_000_000:
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0].sample(n=min(500_000, max(100_000, len(df)//20)), random_state=0)
        df = pd.concat([fraud, legit]).sample(frac=1.0, random_state=0).reset_index(drop=True)
    y = df["isFraud"].astype(int).values
    X = df[["step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFlaggedFraud"]].copy()
    X["errorBalanceOrig"] = X["newbalanceOrig"] + X["amount"] - X["oldbalanceOrg"]
    X["errorBalanceDest"] = X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]
    X["deltaOrig"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
    X["deltaDest"] = X["newbalanceDest"] - X["oldbalanceDest"]
    eps = 1e-6
    X["amt_over_oldOrig"] = X["amount"] / (np.abs(X["oldbalanceOrg"]) + eps)
    X = pd.concat([X.drop(columns=["type"]), pd.get_dummies(X["type"], prefix="type")], axis=1)
    to_scale = [c for c in ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
                            "errorBalanceOrig","errorBalanceDest","deltaOrig","deltaDest","amt_over_oldOrig","step"] if c in X.columns]
    X[to_scale] = StandardScaler().fit_transform(X[to_scale])
    return X.values, y, list(X.columns), "paysim"

def load_dataset_auto(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    df = pd.read_csv(path)
    if "Class" in df.columns: return prepare_creditcard(df)
    if "isFraud" in df.columns: return prepare_paysim(df)
    raise ValueError("Unknown dataset type (need 'Class' or 'isFraud').")

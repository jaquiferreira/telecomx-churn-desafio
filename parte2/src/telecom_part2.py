#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

TARGET_CANDIDATES = ["churn", "Churn", "CHURN", "evadiu", "Evasao", "Exited", "target"]

def find_dataset() -> str:
    """Try common filenames inside data/."""
    candidates = ["telecom_tratado.csv", "dados_tratados.csv"]
    for name in candidates:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Nenhum dataset encontrado em {DATA_DIR}. Coloque 'telecom_tratado.csv' ou 'dados_tratados.csv'."
    )

def detect_target(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col in TARGET_CANDIDATES:
            return col
        if col.lower() in [c.lower() for c in TARGET_CANDIDATES]:
            return col
    # tentativa: achar col binária com nome sugestivo
    for col in df.columns:
        if df[col].nunique() == 2 and any(k in col.lower() for k in ["churn", "evas", "exit", "cancel"]):
            return col
    raise ValueError("Não foi possível detectar a coluna alvo (churn). Renomeie a coluna para 'churn'.")

def to_binary(series: pd.Series) -> pd.Series:
    """Map common yes/no style labels to 1/0; keep numeric as-is."""
    if pd.api.types.is_numeric_dtype(series):
        # Normaliza para {0,1} se for {0,1} já está ok; se for {1,2} mapeia para {0,1}
        unique = sorted(series.dropna().unique().tolist())
        if set(unique) <= set([0,1]):
            return series.astype(int)
        if set(unique) <= set([1,2]):
            return series.replace({1:0, 2:1}).astype(int)
        return series.astype(int, errors="ignore")
    # string-like
    mapping = {
        "yes": 1, "y": 1, "sim": 1, "true": 1, "t": 1, "1": 1,
        "no": 0, "n": 0, "nao": 0, "não": 0, "false": 0, "f": 0, "0": 0
    }
    return series.astype(str).str.strip().str.lower().map(lambda v: mapping.get(v, np.nan)).astype(float)

def split_features(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor

def train_and_eval(X_train, X_test, y_train, y_test, preprocessor) -> Tuple[Dict, Tuple[str, Pipeline]]:
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
    }
    metrics = {}
    best_key = None
    best_score = -np.inf
    best_pipeline = None

    for key, model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        m = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }
        if proba is not None:
            m["roc_auc"] = float(roc_auc_score(y_test, proba))
        else:
            m["roc_auc"] = float("nan")
        metrics[key] = m

        score_for_rank = m.get("roc_auc", -np.inf)
        tie_break = m["f1"]
        if np.isnan(score_for_rank):
            score_for_rank = m["f1"]
            tie_break = m["accuracy"]

        if (score_for_rank > best_score) or (
            np.isclose(score_for_rank, best_score) and tie_break > metrics.get(best_key, {}).get("f1", -np.inf)
        ):
            best_key = key
            best_score = score_for_rank
            best_pipeline = pipe

    return metrics, (best_key, best_pipeline)

def export_feature_importance(pipeline: Pipeline, feature_names: List[str], out_csv: str):
    # Recover transformed feature names
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    # Get feature names from ColumnTransformer
    num_features = pre.transformers_[0][2]
    cat_features = pre.transformers_[1][2]
    cat_encoder = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_output = cat_encoder.get_feature_names_out(cat_features).tolist()
    all_features = list(num_features) + cat_output

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).ravel()
    else:
        # Fallback: zero importances
        importances = np.zeros(len(all_features))

    df_imp = pd.DataFrame({
        "feature": all_features,
        "importance": importances
    }).sort_values("importance", ascending=False)
    df_imp.to_csv(out_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description="Telecom X – Parte 2: Prevendo Churn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    dataset_path = find_dataset()
    print(f"[INFO] Carregando dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    target = detect_target(df)
    print(f"[INFO] Coluna alvo detectada: {target}")

    # binariza alvo
    y_raw = df[target]
    y = to_binary(y_raw)
    if y.isna().any():
        raise ValueError("A coluna alvo contém valores não reconhecidos para binarização. Padronize para 0/1 ou Sim/Não.")
    y = y.astype(int)

    X = df.drop(columns=[target])
    num_cols, cat_cols = split_features(df, target)
    print(f"[INFO] Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

    pre = build_preprocessor(num_cols, cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    metrics, (best_key, best_pipeline) = train_and_eval(X_train, X_test, y_train, y_test, pre)

    # salva métricas
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] Métricas salvas em {metrics_path}")

    # salva importâncias
    imp_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
    export_feature_importance(best_pipeline, X.columns.tolist(), imp_path)
    print(f"[OK] Importâncias salvas em {imp_path}")

    # salva melhor modelo
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"[OK] Modelo salvo em {model_path} (vencedor: {best_key})")

    # relatório simples
    report_md = os.path.join(REPORTS_DIR, "REPORT.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Relatório – Telecom X Parte 2\n\n")
        f.write(f"**Modelo vencedor**: `{best_key}`\n\n")
        f.write("## Métricas (holdout)\n\n")
        f.write("```\n")
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n```\n\n")
        f.write("## Importância de variáveis\n\n")
        f.write("- Ver arquivo `reports/feature_importance.csv`.\n")
    print(f"[OK] Relatório salvo em {report_md}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERRO]", e)
        sys.exit(1)

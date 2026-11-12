from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    ENGINEERED_NUM_COLS,
    ORIGINAL_CAT_COLS,
    ORIGINAL_NUM_COLS,
    apply_imputation_and_features,
    clean_dataframe,
    fit_imputation_stats,
)

COST_FN = 50_000  # Paket terlambat tanpa kabar (Rp50.000)
COST_FP = 5_000   # Pemberitahuan keliru (Rp5.000)


def find_optimal_business_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Mencari threshold pada data validation yang meminimalkan total biaya keluhan."""
    thresholds = np.linspace(0.03, 0.50, 471)
    best = {"threshold": 0.0909, "total_cost": float("inf"), "avg_cost_per_pkg": float("inf")}

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = int(fn * COST_FN + fp * COST_FP)
        if total_cost < best["total_cost"]:
            best = {
                "threshold": round(float(th), 4),
                "total_cost": total_cost,
                "avg_cost_per_pkg": round(total_cost / len(y_true), 2),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
    return best


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw.csv"
    clean_path = root / "data" / "clean.csv"
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pembersihan data mentah & simpan data/clean.csv
    df_raw = pd.read_csv(raw_path)
    df_clean = clean_dataframe(df_raw, deduplicate=True)
    df_clean.drop(columns=["dt"]).to_csv(clean_path, index=False)
    print(f"[1/5] Data bersih disimpan ke {clean_path} ({len(df_clean)} baris unik).")

    # 2. Pembagian data berbasis waktu (Train <= Des 2025, Val Q1 2026, Test Q2 2026)
    train_raw = df_clean[df_clean["dt"] <= "2025-12-31"].copy()
    val_raw = df_clean[(df_clean["dt"] >= "2026-01-01") & (df_clean["dt"] <= "2026-03-31")].copy()
    test_raw = df_clean[df_clean["dt"] >= "2026-04-01"].copy()

    stats = fit_imputation_stats(train_raw)
    train_df = apply_imputation_and_features(train_raw, stats)
    val_df = apply_imputation_and_features(val_raw, stats)
    test_df = apply_imputation_and_features(test_raw, stats)

    # 3. Latih Baseline Wajib: Regresi Logistik dengan fitur asli
    base_prep = ColumnTransformer([
        ("num", StandardScaler(), ORIGINAL_NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ORIGINAL_CAT_COLS),
    ])
    baseline_model = Pipeline([
        ("prep", base_prep),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    baseline_model.fit(train_df[ORIGINAL_NUM_COLS + ORIGINAL_CAT_COLS], train_df["terlambat"])
    val_prob_base = baseline_model.predict_proba(val_df[ORIGINAL_NUM_COLS + ORIGINAL_CAT_COLS])[:, 1]
    val_pr_base = float(average_precision_score(val_df["terlambat"], val_prob_base))
    print(f"[2/5] Baseline Regresi Logistik -> Val PR-AUC: {val_pr_base:.4f}")

    # 4. Latih Model Final: Gradient Boosting + Rekayasa Fitur
    final_features = ENGINEERED_NUM_COLS + ORIGINAL_CAT_COLS
    final_prep = ColumnTransformer([
        ("num", StandardScaler(), ENGINEERED_NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ORIGINAL_CAT_COLS),
    ])
    final_model = Pipeline([
        ("prep", final_prep),
        ("clf", GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )),
    ])
    final_model.fit(train_df[final_features], train_df["terlambat"])
    val_prob_final = final_model.predict_proba(val_df[final_features])[:, 1]
    val_pr_final = float(average_precision_score(val_df["terlambat"], val_prob_final))
    val_roc_final = float(roc_auc_score(val_df["terlambat"], val_prob_final))
    print(f"[3/5] Model Final Gradient Boosting -> Val PR-AUC: {val_pr_final:.4f} (ROC-AUC: {val_roc_final:.4f})")

    # 5. Optimasi Threshold Bisnis pada set Validation
    opt_th = find_optimal_business_threshold(val_df["terlambat"].to_numpy(), val_prob_final)
    th_kabari = opt_th["threshold"]
    th_prioritas = 0.35
    print(f"[4/5] Threshold optimal biaya: {th_kabari} (Total biaya Val: Rp{opt_th['total_cost']:,})")

    # 6. Evaluasi Test Set (DIBUKA SATU KALI DI AKHIR)
    test_prob_base = baseline_model.predict_proba(test_df[ORIGINAL_NUM_COLS + ORIGINAL_CAT_COLS])[:, 1]
    test_pr_base = float(average_precision_score(test_df["terlambat"], test_prob_base))

    test_prob_final = final_model.predict_proba(test_df[final_features])[:, 1]
    test_pr_final = float(average_precision_score(test_df["terlambat"], test_prob_final))
    test_roc_final = float(roc_auc_score(test_df["terlambat"], test_prob_final))

    test_pred = (test_prob_final >= th_kabari).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_df["terlambat"], test_pred).ravel()
    test_cost = int(fn * COST_FN + fp * COST_FP)
    print(f"[5/5] Evaluasi Test Set (1x) -> Baseline PR-AUC: {test_pr_base:.4f} | Final PR-AUC: {test_pr_final:.4f}")

    # 7. Simpan model.joblib & metadata.json
    joblib.dump(final_model, artifacts_dir / "model.joblib")
    metadata = {
        "model_name": "GradientBoostingClassifier + Domain Feature Engineering",
        "baseline_name": "LogisticRegression (Original Features)",
        "features_num": ENGINEERED_NUM_COLS,
        "features_cat": ORIGINAL_CAT_COLS,
        "imputation_stats": stats,
        "thresholds": {
            "kabari_pelanggan": th_kabari,
            "prioritaskan_paket": th_prioritas,
            "theoretical_bayes_threshold": round(COST_FP / (COST_FP + COST_FN), 4),
        },
        "business_costs_idr": {"FN_tanpa_kabar": COST_FN, "FP_alarm_palsu": COST_FP},
        "metrics": {
            "validation": {
                "baseline_pr_auc": round(val_pr_base, 4),
                "final_pr_auc": round(val_pr_final, 4),
                "final_roc_auc": round(val_roc_final, 4),
                "optimal_threshold_summary": opt_th,
            },
            "test": {
                "baseline_pr_auc": round(test_pr_base, 4),
                "final_pr_auc": round(test_pr_final, 4),
                "final_roc_auc": round(test_roc_final, 4),
                "total_cost_idr": test_cost,
                "avg_cost_per_pkg_idr": round(test_cost / len(test_df), 2),
                "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            },
        },
    }
    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("SELESAI! Artifacts tersimpan di folder artifacts/.")


if __name__ == "__main__":
    main()
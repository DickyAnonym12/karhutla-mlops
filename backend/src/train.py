import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Urutan dan nama HARUS sama dengan FEATURES di backend/api/main.py
FEATURES = ["rain_7d", "rain_14d", "rain_30d", "temp",
            "humidity", "wind", "hotspot_7d", "month"]


def evaluate(y, s, thr=0.5):
    pred = (s >= thr).astype(int)
    return {
        "pr_auc": average_precision_score(y, s),
        "roc_auc": roc_auc_score(y, s),
        "precision_at_0.5": precision_score(y, pred, zero_division=0),
        "recall_at_0.5": recall_score(y, pred, zero_division=0),
    }


def build_models(pos_weight):
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=pos_weight, eval_metric="aucpr",
            n_jobs=-1, random_state=42),
    }


def train_all(ds, train_end, eval_start, eval_end, tahap,
              only=None, experiment="karhutla-risiko",
              tracking_uri=None, model_path=None):
    """Latih sampai train_end, evaluasi pada eval_start..eval_end."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    ds = ds.copy()
    ds["tanggal"] = pd.to_datetime(ds["tanggal"])
    tr = ds[ds["tanggal"] <= train_end]
    ev = ds[(ds["tanggal"] >= eval_start) & (ds["tanggal"] <= eval_end)]
    Xtr, ytr, Xev, yev = tr[FEATURES], tr["label"], ev[FEATURES], ev["label"]

    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    baseline = float(yev.mean())
    print(f"[{tahap}] latih {len(tr)} baris, evaluasi {len(ev)} baris, "
          f"baseline PR-AUC {baseline:.3f}")

    rows, best = [], (None, -1.0)
    for name, model in build_models(pos_weight).items():
        if only and name not in only:
            continue
        with mlflow.start_run(run_name=f"{tahap}-{name}"):
            model.fit(Xtr, ytr)
            scores = model.predict_proba(Xev)[:, 1]
            m = evaluate(yev, scores)

            mlflow.set_tag("tahap", tahap)
            mlflow.log_params({"model": name, "train_end": train_end,
                               "eval_start": eval_start, "eval_end": eval_end,
                               "n_train": len(tr), "n_eval": len(ev)})
            mlflow.log_metrics({**m, "baseline_pr_auc": baseline})

            tmp = f"/tmp/{tahap}-{name}.joblib"
            joblib.dump(model, tmp)
            mlflow.log_artifact(tmp)

            rows.append({"model": name, **m, "baseline": baseline})
            if m["pr_auc"] > best[1]:
                best = (model, m["pr_auc"])

    if model_path and best[0] is not None:
        joblib.dump(best[0], model_path)
        print("Model disimpan:", model_path)
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from train import FEATURES, build_models


def psi(ref, cur, bins=10):
    """Population Stability Index satu fitur. <0.1 stabil, 0.1-0.25 sedang, >0.25 besar."""
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return np.nan
    qs[0], qs[-1] = -np.inf, np.inf
    a = np.histogram(ref, qs)[0] / len(ref)
    b = np.histogram(cur, qs)[0] / len(cur)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def metrik_bulan(y, s, thr, n_hari):
    p = s >= thr
    tp = int((p & (y == 1)).sum())
    fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum())
    return {
        "pr_auc": average_precision_score(y, s) if y.sum() > 0 else np.nan,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "alarm_per_hari": (tp + fp) / n_hari,
    }


def simulasi(ds, model_tetap, mulai="2026-01-01", akhir="2026-10-08", thr=0.6):
    d = ds.copy()
    d["tanggal"] = pd.to_datetime(d["tanggal"])
    rows = []
    for bulan in pd.period_range(mulai, akhir, freq="M"):
        awal = bulan.start_time
        tutup = min(bulan.end_time.normalize(), pd.Timestamp(akhir))
        ev = d[(d["tanggal"] >= awal) & (d["tanggal"] <= tutup)]
        if ev.empty:
            continue
        y = ev["label"].values
        n_hari = ev["tanggal"].nunique()

        s_tetap = model_tetap.predict_proba(ev[FEATURES])[:, 1]

        # latih ulang hanya dengan data yang labelnya SUDAH diketahui di awal bulan
        tr = d[d["tanggal"] <= awal - pd.Timedelta(days=8)]
        m_baru = build_models(1)["logreg"].fit(tr[FEATURES], tr["label"])
        s_baru = m_baru.predict_proba(ev[FEATURES])[:, 1]

        a = metrik_bulan(y, s_tetap, thr, n_hari)
        b = metrik_bulan(y, s_baru, thr, n_hari)
        rows.append({
            "bulan": str(bulan), "baseline": y.mean(),
            "prauc_tetap": a["pr_auc"], "prauc_retrain": b["pr_auc"],
            "prec_tetap": a["precision"], "prec_retrain": b["precision"],
            "rec_tetap": a["recall"], "rec_retrain": b["recall"],
            "alarm/hari_tetap": a["alarm_per_hari"],
            "alarm/hari_retrain": b["alarm_per_hari"],
        })
    return pd.DataFrame(rows)


def drift_fitur(ds, ref=("2022-01-08", "2024-12-24"), cur=("2026-01-01", "2026-10-08")):
    d = ds.copy()
    d["tanggal"] = pd.to_datetime(d["tanggal"])
    r = d[(d["tanggal"] >= ref[0]) & (d["tanggal"] <= ref[1])]
    r = r[r["tanggal"].dt.month <= 10]   # samakan bulan dengan 2026 (Jan-Okt)
    c = d[(d["tanggal"] >= cur[0]) & (d["tanggal"] <= cur[1])]
    return pd.Series({f: psi(r[f].values, c[f].values) for f in FEATURES}).round(3)

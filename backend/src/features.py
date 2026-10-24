import geopandas as gpd
import numpy as np
import pandas as pd


def assign_kecamatan(hs, kec, conf_keep=("n", "h")):
    """Saring confidence, lalu petakan tiap hotspot ke kecamatan (spatial join)."""
    hs = hs[hs["confidence"].isin(conf_keep)].copy()
    pts = gpd.GeoDataFrame(
        hs,
        geometry=gpd.points_from_xy(hs["longitude"], hs["latitude"]),
        crs="EPSG:4326",
    )
    poly = kec[["GID_3", "geometry"]].to_crs("EPSG:4326")
    j = gpd.sjoin(pts, poly, how="inner", predicate="within")
    # titik di luar poligon Riau (laut, provinsi tetangga) otomatis terbuang
    return j[["GID_3", "acq_date"]].rename(columns={"GID_3": "gid_3", "acq_date": "tanggal"})


def build_dataset(hs_kec, cu, horizon=7, dropna=True):
    """Satu baris = satu kecamatan pada satu hari.
    Fitur memakai data sampai hari t; label = ada hotspot pada t+1..t+horizon."""
    cu = cu.copy()
    cu["tanggal"] = pd.to_datetime(cu["tanggal"])

    cnt = hs_kec.copy()
    cnt["tanggal"] = pd.to_datetime(cnt["tanggal"])
    cnt = cnt.groupby(["gid_3", "tanggal"]).size().rename("n_hotspot").reset_index()

    df = cu.merge(cnt, on=["gid_3", "tanggal"], how="left")
    df["n_hotspot"] = df["n_hotspot"].fillna(0)
    df = df.sort_values(["gid_3", "tanggal"]).reset_index(drop=True)
    g = df.groupby("gid_3")

    df["rain_7d"] = g["precipitation_sum"].transform(lambda s: s.rolling(7).sum())
    df["rain_14d"] = g["precipitation_sum"].transform(lambda s: s.rolling(14).sum())
    df["rain_30d"] = g["precipitation_sum"].transform(lambda s: s.rolling(30).sum())
    df["hotspot_7d"] = g["n_hotspot"].transform(lambda s: s.rolling(7).sum())

    df["temp"] = df["temperature_2m_mean"]
    df["humidity"] = df["relative_humidity_2m_mean"]
    df["wind"] = df["wind_speed_10m_max"]
    df["month"] = df["tanggal"].dt.month

    # hotspot pada t+1..t+horizon (NaN di ujung periode karena masa depannya belum ada)
    df["future"] = g["n_hotspot"].transform(lambda s: s.rolling(horizon).sum().shift(-horizon))
    df["label"] = np.where(df["future"].isna(), np.nan, (df["future"] > 0).astype(float))

    cols = ["gid_3", "tanggal", "rain_7d", "rain_14d", "rain_30d", "temp",
            "humidity", "wind", "hotspot_7d", "month", "label"]
    out = df[cols]
    if dropna:
        out = out.dropna().reset_index(drop=True)
        out["label"] = out["label"].astype(int)
    return out

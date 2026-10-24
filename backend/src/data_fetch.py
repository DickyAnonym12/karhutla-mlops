import io
import time
from datetime import date, timedelta

import geopandas as gpd
import pandas as pd
import requests

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_kecamatan(path):
    """Baca kecamatan Riau + titik wakil (di dalam poligon) per kecamatan."""
    gdf = gpd.read_file(path)
    pts = gdf.geometry.representative_point()
    gdf = gdf.assign(lat=pts.y, lon=pts.x)
    return gdf


def riau_bbox(gdf, margin=0.1):
    w, s, e, n = gdf.total_bounds
    return (w - margin, s - margin, e + margin, n + margin)


def fetch_firms(map_key, bbox, start, end, source="VIIRS_SNPP_SP", chunk=5, pause=1.0):
    """Ambil hotspot per potongan 'chunk' hari (batas API per permintaan kecil).
    source: VIIRS_SNPP_SP (arsip/standard) atau VIIRS_SNPP_NRT (hampir real-time)."""
    west, south, east, north = bbox
    area = f"{west:.3f},{south:.3f},{east:.3f},{north:.3f}"
    cur, last = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    while cur <= last:
        days = min(chunk, (last - cur).days + 1)
        url = f"{FIRMS_URL}/{map_key}/{source}/{area}/{days}/{cur.date()}"
        r = requests.get(url, timeout=60)
        text = r.text.strip()
        if r.status_code != 200 or not text.startswith("latitude"):
            # pesan error dipotong supaya MAP_KEY tidak ikut tercetak
            print("Peringatan", cur.date(), r.status_code, text[:80].replace(map_key, "***"))
        else:
            frames.append(pd.read_csv(io.StringIO(text)))
        cur += timedelta(days=days)
        time.sleep(pause)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return df


def fetch_weather(kec, start, end, batch=10, pause=2.0):
    """Cuaca harian per titik kecamatan dari Open-Meteo (beberapa titik per permintaan)."""
    daily = ("precipitation_sum,temperature_2m_mean,"
             "relative_humidity_2m_mean,wind_speed_10m_max")
    rows = []
    kec = kec.reset_index(drop=True)
    for i in range(0, len(kec), batch):
        sub = kec.iloc[i:i + batch]
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in sub["lat"]),
            "longitude": ",".join(f"{v:.4f}" for v in sub["lon"]),
            "start_date": start, "end_date": end,
            "daily": daily, "timezone": "Asia/Jakarta",
        }
        r = requests.get(METEO_URL, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        data = data if isinstance(data, list) else [data]
        for (_, k), d in zip(sub.iterrows(), data):
            df = pd.DataFrame(d["daily"]).rename(columns={"time": "tanggal"})
            df.insert(0, "gid_3", k["GID_3"])
            rows.append(df)
        time.sleep(pause)
    return pd.concat(rows, ignore_index=True)


def fetch_weather_cached(kec, start, end, cache_dir, batch=5, grid=0.25, pause=5):
    """Cuaca per titik grid unik, disimpan per potongan, menunggu saat kena 429."""
    import os
    os.makedirs(cache_dir, exist_ok=True)

    k = kec[["GID_3", "lat", "lon"]].copy()
    k["glat"] = (k["lat"] / grid).round() * grid
    k["glon"] = (k["lon"] / grid).round() * grid
    cells = k[["glat", "glon"]].drop_duplicates().reset_index(drop=True)
    cells["cell"] = cells.index
    k = k.merge(cells, on=["glat", "glon"])
    print(len(k), "kecamatan ->", len(cells), "titik grid unik")

    daily = ("precipitation_sum,temperature_2m_mean,"
             "relative_humidity_2m_mean,wind_speed_10m_max")
    waits = [70, 300, 900]   # detik: batas menit, lalu batas lebih lama
    parts = []

    for i in range(0, len(cells), batch):
        f = f"{cache_dir}/cu_{start}_{end}_{i}.csv"
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
            continue

        sub = cells.iloc[i:i + batch]
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in sub["glat"]),
            "longitude": ",".join(f"{v:.4f}" for v in sub["glon"]),
            "start_date": start, "end_date": end,
            "daily": daily, "timezone": "Asia/Jakarta",
        }
        for attempt in range(len(waits) + 1):
            r = requests.get(METEO_URL, params=params, timeout=180)
            if r.status_code != 429:
                break
            if attempt == len(waits):
                raise RuntimeError(
                    f"Masih 429 di potongan {i}. Kemungkinan batas jam/hari tercapai. "
                    "Hasil yang sudah ada tersimpan; jalankan ulang nanti.")
            print("429 di potongan", i, "- tunggu", waits[attempt], "detik")
            time.sleep(waits[attempt])
        r.raise_for_status()

        data = r.json()
        data = data if isinstance(data, list) else [data]
        rows = []
        for (_, c), d in zip(sub.iterrows(), data):
            df = pd.DataFrame(d["daily"]).rename(columns={"time": "tanggal"})
            df.insert(0, "cell", c["cell"])
            rows.append(df)
        part = pd.concat(rows, ignore_index=True)
        part.to_csv(f, index=False)
        parts.append(part)
        time.sleep(pause)

    cu_cells = pd.concat(parts, ignore_index=True)
    out = (k[["GID_3", "cell"]].rename(columns={"GID_3": "gid_3"})
             .merge(cu_cells, on="cell").drop(columns="cell"))
    return out

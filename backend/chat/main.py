import difflib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE = os.getenv("LM_BASE", "http://localhost:1234/v1")
MODEL = os.getenv("LM_MODEL", "qwen2.5-vl-7b-instruct")
ROOT = Path(__file__).resolve().parents[2]

DF = pd.read_csv(ROOT / "data" / "prediksi_demo.csv")
TANGGAL = str(DF["tanggal"].iloc[0])


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


DF["k_norm"] = DF["kecamatan"].map(norm)
DF["b_norm"] = DF["kabupaten"].map(norm)
KEC = DF["k_norm"].tolist()
KAB = sorted(DF["b_norm"].unique())
BOLEH = {float(i) for i in range(1, 11)}

SYS_INTENT = """Kamu pengurai pertanyaan petugas BPBD tentang risiko karhutla di Riau.
Balas HANYA satu objek JSON, tanpa teks lain dan tanpa markdown.
Format: {"intent": "risiko" atau "top" atau "lainnya", "kecamatan": teks atau null, "n": angka atau null}
- "risiko": menanyakan risiko satu kecamatan; isi "kecamatan" persis seperti ditulis pengguna
- "top": menanyakan kecamatan paling berisiko; isi "n" (bawaan 5)
- "lainnya": pertanyaan di luar topik"""

SYS_JAWAB = """Kamu asisten untuk petugas BPBD. Tulis jawaban singkat (maksimal 3 kalimat) dalam bahasa Indonesia berdasarkan DATA.
Aturan: gunakan HANYA nama dan angka yang ada di DATA, tulis angka persis seperti di DATA; jangan menambah angka, prediksi, atau saran tindakan lapangan; untuk hotspot tulis jumlah titik dalam 7 hari terakhir persis seperti di DATA dan jangan menulis hotspot selama 7 hari; sebut tanggal data; gunakan istilah 'skor risiko' (bukan peluang kebakaran)."""


def tanya_llm(messages, max_tokens=300):
    r = requests.post(f"{BASE}/chat/completions", timeout=120, json={
        "model": MODEL, "temperature": 0, "max_tokens": max_tokens,
        "messages": messages})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def cadangan_intent(pesan):
    p = norm(pesan)
    if re.search(r"\b(top|teratas|paling)\b", p):
        m = re.search(r"\b(\d+)\b", p)
        return {"intent": "top", "n": int(m.group(1)) if m else 5}
    for nama in sorted(KEC + KAB, key=len, reverse=True):
        if re.search(r"\b" + re.escape(nama) + r"\b", p):
            return {"intent": "risiko", "kecamatan": nama}
    return {"intent": "lainnya"}


def intent(pesan):
    try:
        teks = tanya_llm([{"role": "system", "content": SYS_INTENT},
                          {"role": "user", "content": pesan}], max_tokens=80)
        teks = re.sub(r"^```(?:json)?|```$", "", teks, flags=re.M).strip()
        d = json.loads(teks)
        if isinstance(d, dict) and d.get("intent") in ("risiko", "top", "lainnya"):
            return d
    except Exception:
        pass
    return cadangan_intent(pesan)


def cari(nama, pesan):
    n = re.sub(r"^(kecamatan|kec|kabupaten|kab) ", "", norm(nama or ""))
    p = pesan.lower()
    mau_kab = bool(re.search(r"\bkab(upaten)?\b", p))
    mau_kec = bool(re.search(r"\bkec(amatan)?\b", p))
    ada_kec, ada_kab = n in KEC, n in KAB
    if ada_kec and ada_kab and not (mau_kab or mau_kec):
        return "ambigu", n
    if ada_kab and (mau_kab or not ada_kec):
        return "kab", DF[DF["b_norm"] == n]
    if ada_kec:
        return "kec", DF[DF["k_norm"] == n]
    return "tidak", difflib.get_close_matches(n, KEC + KAB, n=3, cutoff=0.7)


def nama_dalam(pesan):
    p = " " + norm(pesan) + " "
    ketemu = []
    for nama in sorted(set(KEC + KAB), key=len, reverse=True):
        pola = " " + nama + " "
        if pola in p:
            ketemu.append(nama)
            p = p.replace(pola, " ", 1)
    return ketemu


def level_cocok(teks, rows):
    t = teks.lower()
    if re.search(r"peluang|probabilitas|kemungkinan|persen|%|tingkat kebakaran|akan terbakar", t):
        return False
    if len(rows) != 1:
        return True
    ada = set(re.findall(r"\b(tinggi|sedang|rendah)\b", t))
    return ada <= {str(rows.iloc[0]["level"]).lower()}


def angka(s):
    return {float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", s)}


def teks_baris(r):
    return (f"{r.kecamatan} ({r.kabupaten}): skor risiko {r.skor:.3f}, level {r.level}, "
            f"hujan 7 hari {r.rain_7d:.1f} mm, hujan 30 hari {r.rain_30d:.1f} mm, "
            f"jumlah titik hotspot dalam 7 hari terakhir: {int(r.hotspot_7d)}")


def kartu(rows):
    return [{"kecamatan": r.kecamatan, "kabupaten": r.kabupaten, "skor": round(float(r.skor), 3),
             "level": r.level, "hotspot_7d": int(r.hotspot_7d)} for r in rows.itertuples()]


def jawab(pesan, rows, ringkas, cadangan):
    konteks = ringkas + "\n" + "\n".join(teks_baris(r) for r in rows.itertuples())
    try:
        teks = tanya_llm([
            {"role": "system", "content": SYS_JAWAB},
            {"role": "user", "content": f"Pertanyaan: {pesan}\nTanggal data: {TANGGAL}\nDATA:\n{konteks}"}],
            max_tokens=250)
    except Exception:
        return cadangan, "templat"
    if teks and level_cocok(teks, rows) and angka(teks) <= (angka(konteks + " " + pesan + " " + TANGGAL) | BOLEH):
        return teks, "llm"
    return cadangan, "templat"


def catat(pesan, hasil):
    try:
        with open(ROOT / "data" / "log_chat.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"waktu": datetime.now().isoformat(timespec="seconds"),
                                "pesan": pesan, **{k: hasil[k] for k in ("jawaban", "sumber")}},
                               ensure_ascii=False) + "\n")
    except Exception:
        pass


app = FastAPI(title="Asisten Risiko Karhutla")


class Chat(BaseModel):
    pesan: str


def selesai(pesan, jawaban, sumber, rows=None):
    hasil = {"jawaban": jawaban, "sumber": sumber, "tanggal": TANGGAL,
             "kartu": kartu(rows) if rows is not None else []}
    catat(pesan, hasil)
    return hasil


@app.get("/health")
def health():
    try:
        ok = requests.get(f"{BASE}/models", timeout=3).ok
    except Exception:
        ok = False
    return {"status": "ok", "baris": len(DF), "tanggal": TANGGAL, "llm_aktif": ok}


@app.post("/chat")
def chat(req: Chat):
    pesan = req.pesan.strip()[:300]
    if not pesan:
        return selesai(pesan, "Silakan ketik pertanyaan.", "tetap")
    nama_banyak = nama_dalam(pesan)
    if len(nama_banyak) >= 2:
        baris = []
        for nm in nama_banyak:
            jenis, h = cari(nm, pesan)
            if jenis == "ambigu":
                return selesai(pesan, f"'{h.title()}' adalah nama kecamatan sekaligus kabupaten. "
                               f"Tulis 'kecamatan {h.title()}' atau 'kabupaten {h.title()}'.", "tetap")
            if jenis != "kec":
                return selesai(pesan, "Untuk membandingkan, sebut nama kecamatan (bukan kabupaten).", "tetap")
            baris.append(h.iloc[0])
        rows = pd.DataFrame(baris).sort_values("skor", ascending=False)
        daftar = "; ".join(f"{i}. {r.kecamatan} ({r.skor:.3f}, {r.level})"
                           for i, r in enumerate(rows.itertuples(), 1))
        cad = f"Per {TANGGAL}, urutan skor risiko: {daftar}."
        if len(rows) == 2:
            a, b = rows.iloc[0], rows.iloc[1]
            cad += (f" Skor risiko {a.kecamatan} lebih tinggi dari {b.kecamatan}."
                    if a.skor > b.skor else " Skor risiko keduanya sama.")
        return selesai(pesan, cad, "templat", rows)
    it = intent(pesan)

    if it["intent"] == "top":
        try:
            n = max(1, min(int(it.get("n") or 5), 10))
        except Exception:
            n = 5
        naik = bool(re.search(r"\b(aman|terendah|rendah|sedikit|terkecil)\b", norm(pesan)))
        kata = "terendah" if naik else "tertinggi"
        rows = DF.nsmallest(n, "skor") if naik else DF.nlargest(n, "skor")
        daftar = "; ".join(f"{i}. {r.kecamatan} ({r.skor:.3f}, {r.level})"
                           for i, r in enumerate(rows.itertuples(), 1))
        cad = f"Per {TANGGAL}, {n} kecamatan dengan skor risiko {kata}: {daftar}."
        return selesai(pesan, cad, "templat", rows)

    if it["intent"] == "risiko":
        jenis, hasil = cari(it.get("kecamatan"), pesan)
        if jenis == "tidak":
            alt = cadangan_intent(pesan)
            if alt.get("intent") == "risiko":
                jenis, hasil = cari(alt.get("kecamatan"), pesan)
        if jenis == "ambigu":
            return selesai(pesan, f"'{hasil.title()}' adalah nama kecamatan sekaligus kabupaten. "
                           f"Tulis 'kecamatan {hasil.title()}' atau 'kabupaten {hasil.title()}'.", "tetap")
        if jenis == "tidak":
            saran = ", ".join(s.title() for s in hasil)
            return selesai(pesan, "Nama itu tidak ditemukan." + (f" Maksud Anda: {saran}?" if saran else ""), "tetap")
        if jenis == "kec":
            r = hasil.iloc[0]
            cad = f"Per {TANGGAL}, {r.kecamatan} ({r.kabupaten}) memiliki skor risiko {r.skor:.3f} (level {r.level})."
            teks, sumber = jawab(pesan, hasil, f"Risiko satu kecamatan: {r.kecamatan}.", cad)
            return selesai(pesan, teks, sumber, hasil)
        sub = hasil.nlargest(5, "skor")
        c = hasil["level"].value_counts()
        a, b, d = int(c.get("TINGGI", 0)), int(c.get("SEDANG", 0)), int(c.get("RENDAH", 0))
        kab = hasil["kabupaten"].iloc[0]
        r = sub.iloc[0]
        ringkas = f"Kabupaten {kab}: {len(hasil)} kecamatan, TINGGI {a}, SEDANG {b}, RENDAH {d}. Lima skor tertinggi:"
        cad = (f"Per {TANGGAL}, di Kabupaten {kab}: {a} kecamatan TINGGI, {b} SEDANG, {d} RENDAH. "
               f"Tertinggi: {r.kecamatan} ({r.skor:.3f}).")
        return selesai(pesan, cad, "templat", sub)

    return selesai(pesan, "Saya hanya menjawab risiko karhutla di Riau. Coba tanya: 'risiko di Minas' "
                   "atau '5 kecamatan paling berisiko'.", "tetap")


@app.get("/", response_class=HTMLResponse)
def beranda():
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

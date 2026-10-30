import json, re, time, requests

BASE = "http://localhost:1234/v1"
MODEL = "qwen2.5-vl-7b-instruct"   # ganti dengan model lain untuk membandingkan

SYSTEM = """Kamu pengurai pertanyaan petugas BPBD tentang risiko karhutla di Riau.
Balas HANYA satu objek JSON, tanpa teks lain dan tanpa markdown.
Format: {"intent": "risiko" atau "top" atau "lainnya", "kecamatan": teks atau null, "n": angka atau null}
- "risiko": menanyakan risiko satu kecamatan; isi "kecamatan" persis seperti ditulis pengguna
- "top": menanyakan kecamatan paling berisiko; isi "n" (bawaan 5)
- "lainnya": pertanyaan di luar topik"""

TES = [
    "Besok Siak aman nggak?",
    "kecamatan mana yang paling rawan minggu ini?",
    "tolong 3 kecamatan teratas",
    "Gimana risiko di Minas?",
    "rengat barat gmn",
    "risiko di Pangkalan Kerinci hari ini?",
    "resep nasi goreng enak",
]

for q in TES:
    t = time.time()
    r = requests.post(f"{BASE}/chat/completions", timeout=180, json={
        "model": MODEL, "temperature": 0,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": q}],
    })
    teks = r.json()["choices"][0]["message"]["content"].strip()
    bersih = re.sub(r"^```(?:json)?|```$", "", teks, flags=re.M).strip()
    try:
        hasil, ok = json.loads(bersih), "JSON OK"
    except Exception:
        hasil, ok = bersih, "JSON GAGAL"
    print(f"[{ok}] {time.time()-t:.1f} dtk | {q}\n    -> {hasil}")

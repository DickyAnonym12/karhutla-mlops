import requests, time

TES = ["3 kecamatan paling aman", "5 kecamatan dengan skor terendah",
       "apakah Minas lebih rawan dari Rengat?", "risiko di Minas dan Siak",
       "bandingkan kecamatan Minas dan kecamatan Siak", "bandingkan Minas dan Rengat Barat",
       "kabupaten Pelalawan", "5 kecamatan paling berisiko", "kecamatan Siak", "resep nasi goreng"]

for q in TES:
    t = time.time()
    d = requests.post("http://localhost:8000/chat", json={"pesan": q}, timeout=180).json()
    print(f"\n[{time.time()-t:.1f} dtk | sumber: {d['sumber']}] {q}")
    print("  ->", d["jawaban"])
    print("  kartu:", [(k["kecamatan"], k["skor"], k["level"]) for k in d["kartu"]])

# Sistem Prediksi Risiko Karhutla Berbasis MLOps

Memprediksi risiko kebakaran hutan dan lahan (karhutla) per kecamatan
di Riau untuk H+1 sampai H+7, lalu mengirim peringatan ke petugas
BPBD melalui Telegram.

## Masalah
Pemantauan yang ada (SiPongi, hotspot BMKG) bersifat reaktif: titik
panas terdeteksi setelah api muncul.

## Solusi
Model klasifikasi risiko berbasis cuaca dan riwayat hotspot, dibangun
dengan pendekatan MLOps (pipeline otomatis, monitoring, retraining).

## Alur
Schedule Trigger (n8n) -> tarik FIRMS + cuaca -> fitur per kecamatan
-> FastAPI /predict -> simpan prediksi -> kirim ke Telegram.

## Teknologi
n8n, FastAPI, MLflow, Docker, Telegram Bot API, scikit-learn/XGBoost.

## Sumber data
- NASA FIRMS (hotspot VIIRS)
- Open-Meteo (cuaca)
- GADM (batas kecamatan)

## Menjalankan
1. Salin `.env.example` menjadi `.env`, lalu isi MAP_KEY dan token Telegram
2. `docker compose up --build`

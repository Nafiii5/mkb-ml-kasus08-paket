# Prediksi Keterlambatan Paket Hub Bandung · Individu

Kasus: Kasus 08 — Keterlambatan Paket · Aplikasi: (https://mkb-ml-kasus08-paket-4w5cwkzcleolhe57kccyvu.streamlit.app/) · Video: [ISI TAUTAN VIDEO DEMO ANDA]

## Anggota
| Nama | NIM | Bagian yang dikerjakan |
| :--- | :--- | :--- |
| Nafira Zahra Anantha | 2405832 | [misalnya EDA dan pembersihan data] |
| Nailah Khoirun Widad | 2401678 | [misalnya EDA dan pembersihan data] |

## Masalah dan pengguna
Pengguna utama sistem ini adalah **staf operasional *hub* Bandung** dan **staf layanan pelanggan (*customer service*)** saat paket baru dijemput (*pickup*). Model ini membantu memutuskan apakah paket cukup diproses secara reguler, perlu diberi pemberitahuan dini kepada pelanggan terkait potensi keterlambatan, atau harus diprioritaskan pada proses sortir dan pemberangkatan armada.

## Data dan pembersihan
| Masalah yang ditemukan | Jumlah baris | Cara menangani |
| :--- | :--- | :--- |
| Baris duplikat pada id_resi | 500 | Dihapus dengan drop_duplicates pada id_resi sehingga tersisa 40.000 resi unik |
| Nilai jarak_km kosong (NaN) | 813 | Diisi menggunakan median jarak_km per kota_tujuan yang dihitung hanya dari data *train* |
| Nilai cuaca_saat_pickup kosong (NaN) | 406 | Diisi dengan kategori eksplisit 	idak_diketahui |
| Nilai Berat_kg tidak valid (<= 0) | 81 | Dibuat indikator erat_hilang_flag = 1, diubah menjadi NaN, lalu diisi median erat_kg data *train* |
| Nilai Berat_kg dalam satuan gram (>= 100) | 1,208 | Dikonversi ke kilogram dengan membagi nilai dengan 1.000 |
| Inkonsistensi penulisan kota_tujuan & layanan | 40.000 | Dinormalisasi dari 28 variasi kota menjadi 15 kota baku, serta 7 variasi layanan (EXP ke EXPRESS) |

## Hasil
| Model | PR-AUC validation | PR-AUC test |
| :--- | :--- | :--- |
| Baseline: Logistic Regression (fitur asli) | 0.6455 | 0.3563 |
| Final: Gradient Boosting + Rekayasa Fitur Domain | 0.6556 | 0.3665 |

**Threshold yang dipakai:** **0.111** (notifikasi pelanggan) dan **0.35** (prioritas sortir), karena biaya asimetris bisnis di mana paket terlambat tanpa kabar menimbulkan biaya keluhan **Rp50.000** (10x lebih mahal dari notifikasi keliru **Rp5.000**).

## Cara menjalankan
`ash
pip install -r requirements.txt
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
`

## Struktur repositori
`	ext
notebooks/      notebook dari EDA sampai penyimpanan model
data/           data mentah (raw.csv) dan data bersih (clean.csv)
src/features.py fungsi pembuat fitur, dipakai saat pelatihan dan saat prediksi
artifacts/      model.joblib + metadata.json
api/            API (FastAPI)
app/            antarmuka (Streamlit)
`

## Keterbatasan
- **Gangguan mendadak di luar histori:** Model tidak dapat memprediksi keterlambatan akibat kejadian luar biasa seperti kecelakaan armada, bencana alam/banjir yang memutus jalur antarprovinsi, atau mogok kerja.
- **Data operasional lanjutan belum tersedia:** Prediksi hanya memakai data saat *pickup*; antrean sortir per *shift*, kapasitas truk saat keberangkatan, dan performa kurir *last-mile* belum tercakup.
- **Pergeseran distribusi musiman:** Lonjakan volume ekstrem atau rute baru dapat menggeser kalibrasi probabilitas sehingga model perlu dilatih ulang secara berkala.

## Daftar periksa sebelum dikumpulkan
- [x] Repositori bersifat publik dan memiliki riwayat commit
- [x] Notebook dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [x] Baseline wajib dilaporkan dan dibandingkan
- [x] Data test hanya dibuka satu kali, di akhir
- [x] artifacts/metadata.json berisi versi model, fitur masukan, threshold, dan metrik
- [x] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [x] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit

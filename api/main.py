from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import apply_imputation_and_features, clean_dataframe

MODEL = joblib.load(ROOT / "artifacts" / "model.joblib")
with open(ROOT / "artifacts" / "metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

app = FastAPI(
    title="API Prediksi Keterlambatan Paket - Hub Bandung",
    description="Memprediksi probabilitas keterlambatan paket saat dijemput beserta rekomendasi tindakan.",
    version="1.0.0",
)


class PaketInput(BaseModel):
    tanggal_pickup: date = Field(..., description="Tanggal paket dijemput (YYYY-MM-DD)")
    jam_pickup: int = Field(..., ge=7, le=22, description="Jam operasional pickup (7–22)")
    kota_tujuan: str = Field(..., min_length=2, max_length=50, description="Kota tujuan pengiriman")
    jarak_km: float | None = Field(None, gt=0, le=5000, description="Jarak dari Bandung dalam km (opsional)")
    jumlah_transit: int = Field(..., ge=1, le=10, description="Jumlah hub transit (1–10)")
    layanan: Literal["EXPRESS", "REG", "HEMAT"] = Field(..., description="Jenis layanan")
    janji_hari: int = Field(..., ge=1, le=14, description="Janji SLA tiba dalam hari (1–14)")
    berat_kg: float = Field(..., gt=0, le=100, description="Berat paket dalam kg (0.01–100)")
    metode_bayar: Literal["COD", "non_COD"] = Field(..., description="Metode pembayaran")
    cuaca_saat_pickup: Literal["cerah", "berawan", "hujan", "hujan_lebat"] = Field(
        ..., description="Cuaca saat pickup"
    )
    volume_hub_hari: int = Field(..., ge=100, le=50000, description="Volume paket masuk hub hari ini")

    @field_validator("kota_tujuan")
    @classmethod
    def validate_kota(cls, v: str) -> str:
        if not v.strip() or v.strip().isdigit():
            raise ValueError("Nama kota_tujuan tidak boleh kosong atau hanya angka.")
        return v.strip()


class PrediksiOutput(BaseModel):
    probabilitas_terlambat: float
    prediksi_terlambat: int
    status_risiko: str
    rekomendasi: str
    threshold_kabari: float
    threshold_prioritas: float
    faktor_risiko_utama: list[str]


def deteksi_faktor_risiko(row: pd.Series) -> list[str]:
    faktor = []
    if row["volume_hub_hari"] > META["imputation_stats"]["vol_q75"]:
        faktor.append(f"Volume hub tinggi ({int(row['volume_hub_hari']):,} paket/hari)")
    if row["jam_pickup"] >= 17:
        faktor.append(f"Pickup sore/malam pukul {int(row['jam_pickup'])}:00 (melewati cutoff armada)")
    if row["kecepatan_wajib_km_per_hari"] > 250:
        faktor.append(f"Target jarak harian ketat ({row['kecepatan_wajib_km_per_hari']:.0f} km/hari)")
    if row["jumlah_transit"] >= 3:
        faktor.append(f"Banyak titik transit ({int(row['jumlah_transit'])} hub)")
    if row["cuaca_saat_pickup"] in ("hujan", "hujan_lebat"):
        faktor.append(f"Cuaca kurang mendukung ({row['cuaca_saat_pickup']})")
    return faktor or ["Kondisi operasional relatif normal"]


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "model": META["model_name"]}


@app.post("/predict", response_model=PrediksiOutput)
def predict_delay(paket: PaketInput) -> PrediksiOutput:
    try:
        raw_df = pd.DataFrame([paket.model_dump()])
        raw_df["tanggal_pickup"] = raw_df["tanggal_pickup"].astype(str)

        cleaned = clean_dataframe(raw_df, deduplicate=False)
        featured = apply_imputation_and_features(cleaned, META["imputation_stats"])

        cols = META["features_num"] + META["features_cat"]
        prob = float(MODEL.predict_proba(featured[cols])[0, 1])

        th_kabari = META["thresholds"]["kabari_pelanggan"]
        th_prioritas = META["thresholds"]["prioritaskan_paket"]

        if prob >= th_prioritas:
            status = "RISIKO TINGGI"
            rekomendasi = "PRIORITASKAN SORTIR & KABARI PELANGGAN LEBIH AWAL"
        elif prob >= th_kabari:
            status = "RISIKO SEDANG"
            rekomendasi = "KABARI PELANGGAN POTENSI KETERLAMBATAN"
        else:
            status = "AMAN"
            rekomendasi = "PROSES REGULER (AMAN)"

        return PrediksiOutput(
            probabilitas_terlambat=round(prob, 4),
            prediksi_terlambat=int(prob >= th_kabari),
            status_risiko=status,
            rekomendasi=rekomendasi,
            threshold_kabari=th_kabari,
            threshold_prioritas=th_prioritas,
            faktor_risiko_utama=deteksi_faktor_risiko(featured.iloc[0]),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal memproses prediksi: {exc}") from exc
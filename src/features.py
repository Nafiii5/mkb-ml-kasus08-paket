from __future__ import annotations
import numpy as np
import pandas as pd

KOTA_MAP = {
    "jakarta": "Jakarta", "jkt": "Jakarta", "dki jakarta": "Jakarta",
    "bekasi": "Bekasi", "kota bekasi": "Bekasi",
    "surabaya": "Surabaya", "sby": "Surabaya",
    "yogyakarta": "Yogyakarta", "di yogyakarta": "Yogyakarta", "jogja": "Yogyakarta", "yogya": "Yogyakarta",
    "denpasar": "Denpasar", "bali": "Denpasar",
    "medan": "Medan",
    "bogor": "Bogor",
    "semarang": "Semarang",
    "tasikmalaya": "Tasikmalaya",
    "cirebon": "Cirebon",
    "malang": "Malang",
    "makassar": "Makassar",
    "palembang": "Palembang",
    "balikpapan": "Balikpapan",
    "pontianak": "Pontianak",
}

LAYANAN_MAP = {
    "REG": "REG",
    "HEMAT": "HEMAT",
    "EXPRESS": "EXPRESS",
    "EXP": "EXPRESS",
}

ORIGINAL_NUM_COLS = [
    "jam_pickup",
    "jarak_km",
    "jumlah_transit",
    "janji_hari",
    "berat_kg",
    "volume_hub_hari",
]

ORIGINAL_CAT_COLS = [
    "kota_tujuan",
    "layanan",
    "metode_bayar",
    "cuaca_saat_pickup",
]

ENGINEERED_NUM_COLS = ORIGINAL_NUM_COLS + [
    "berat_hilang_flag",
    "hari_minggu",
    "is_akhir_pekan",
    "tanggal_bulan",
    "is_tanggal_kembar",
    "is_akhir_bulan",
    "kecepatan_wajib_km_per_hari",
    "jarak_per_transit",
    "transit_per_janji_hari",
    "is_pickup_sore_malam",
    "is_pickup_malam",
    "rasio_volume_hub",
    "is_hub_padat",
    "is_hujan_lebat",
    "is_hujan_any",
    "is_cod",
    "skor_beban_rute",
    "interaksi_padat_malam",
    "interaksi_cuaca_transit",
    "log_berat_kg",
]


def clean_dataframe(df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
    """Membersihkan duplikat resi, variasi teks kategori, nilai -1, dan satuan gram pada berat_kg."""
    out = df.copy()
    if deduplicate and "id_resi" in out.columns:
        out = out.drop_duplicates(subset=["id_resi"], keep="first").reset_index(drop=True)

    out["kota_tujuan"] = (
        out["kota_tujuan"].astype(str).str.strip().str.lower().map(KOTA_MAP).fillna("Lainnya")
    )
    out["layanan"] = (
        out["layanan"].astype(str).str.strip().str.upper().map(LAYANAN_MAP).fillna("REG")
    )
    out["metode_bayar"] = out["metode_bayar"].astype(str).str.strip()
    out["cuaca_saat_pickup"] = (
        out["cuaca_saat_pickup"].fillna("tidak_diketahui").astype(str).str.strip().str.lower()
    )

    out["berat_kg"] = pd.to_numeric(out["berat_kg"], errors="coerce")
    out["berat_hilang_flag"] = (out["berat_kg"].isna() | (out["berat_kg"] <= 0)).astype(int)
    out.loc[out["berat_kg"] <= 0, "berat_kg"] = np.nan
    out.loc[out["berat_kg"] >= 100, "berat_kg"] = out.loc[out["berat_kg"] >= 100, "berat_kg"] / 1000.0

    out["jarak_km"] = pd.to_numeric(out["jarak_km"], errors="coerce")
    out["dt"] = pd.to_datetime(out["tanggal_pickup"])
    return out


def fit_imputation_stats(train_df: pd.DataFrame) -> dict:
    """Menghitung median imputasi dan kuantil referensi hanya dari data train (mencegah leakage)."""
    return {
        "kota_jarak_median": train_df.groupby("kota_tujuan")["jarak_km"].median().to_dict(),
        "global_jarak_median": float(train_df["jarak_km"].median()),
        "global_berat_median": float(train_df["berat_kg"].median()),
        "vol_median": float(train_df["volume_hub_hari"].median()),
        "vol_q75": float(train_df["volume_hub_hari"].quantile(0.75)),
    }


def apply_imputation_and_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Mengisi nilai kosong dan membentuk fitur domain logistik saat penjemputan."""
    out = df.copy()
    if "dt" not in out.columns:
        out["dt"] = pd.to_datetime(out["tanggal_pickup"])

    mapped_jarak = out["kota_tujuan"].map(stats["kota_jarak_median"])
    out["jarak_km"] = out["jarak_km"].fillna(mapped_jarak).fillna(stats["global_jarak_median"])
    out["berat_kg"] = out["berat_kg"].fillna(stats["global_berat_median"])

    # Fitur kalender saat penjemputan
    out["hari_minggu"] = out["dt"].dt.dayofweek
    out["is_akhir_pekan"] = (out["hari_minggu"] >= 5).astype(int)
    out["tanggal_bulan"] = out["dt"].dt.day
    out["is_tanggal_kembar"] = (out["dt"].dt.month == out["dt"].dt.day).astype(int)
    out["is_akhir_bulan"] = (out["dt"].dt.day >= 25).astype(int)

    # Fitur tekanan operasional & interaksi non-linear
    out["kecepatan_wajib_km_per_hari"] = out["jarak_km"] / np.maximum(out["janji_hari"], 1)
    out["jarak_per_transit"] = out["jarak_km"] / np.maximum(out["jumlah_transit"], 1)
    out["transit_per_janji_hari"] = out["jumlah_transit"] / np.maximum(out["janji_hari"], 1)
    out["is_pickup_sore_malam"] = (out["jam_pickup"] >= 17).astype(int)
    out["is_pickup_malam"] = (out["jam_pickup"] >= 19).astype(int)
    out["rasio_volume_hub"] = out["volume_hub_hari"] / stats["vol_median"]
    out["is_hub_padat"] = (out["volume_hub_hari"] > stats["vol_q75"]).astype(int)
    out["is_hujan_lebat"] = (out["cuaca_saat_pickup"] == "hujan_lebat").astype(int)
    out["is_hujan_any"] = out["cuaca_saat_pickup"].isin(["hujan", "hujan_lebat"]).astype(int)
    out["is_cod"] = (out["metode_bayar"] == "COD").astype(int)

    out["skor_beban_rute"] = out["kecepatan_wajib_km_per_hari"] * out["jumlah_transit"]
    out["interaksi_padat_malam"] = out["rasio_volume_hub"] * out["jam_pickup"]
    out["interaksi_cuaca_transit"] = (1 + out["is_hujan_any"] + out["is_hujan_lebat"]) * out["jumlah_transit"]
    out["log_berat_kg"] = np.log1p(out["berat_kg"])
    return out
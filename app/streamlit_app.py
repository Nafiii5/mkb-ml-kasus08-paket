from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import apply_imputation_and_features, clean_dataframe

st.set_page_config(
    page_title="Prediksi Keterlambatan Paket · Hub Bandung",
    page_icon="📦",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    with open(ROOT / "artifacts" / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


MODEL, META = load_artifacts()
KOTA_PILIHAN = sorted(META["imputation_stats"]["kota_jarak_median"].keys())


def validasi_input(
    kota: str,
    jam_pickup: int,
    jarak_str: str,
    berat_kg: float,
    janji_hari: int,
    jumlah_transit: int,
    volume_hub: int,
) -> tuple[list[str], float | None]:
    """Memeriksa masukan tidak wajar (kosong, mustahil, atau di luar rentang operasional)."""
    errors: list[str] = []

    if not kota or not kota.strip():
        errors.append("Kota tujuan tidak boleh kosong.")
    if not (7 <= jam_pickup <= 22):
        errors.append(f"Jam pickup ({jam_pickup}:00) di luar jam operasional hub (07:00–22:00).")
    if berat_kg <= 0 or berat_kg > 100:
        errors.append(f"Berat paket ({berat_kg} kg) tidak wajar. Rentang valid: 0,01–100 kg.")
    if janji_hari < 1 or janji_hari > 14:
        errors.append(f"Janji SLA ({janji_hari} hari) di luar rentang valid (1–14 hari).")
    if jumlah_transit < 1 or jumlah_transit > 10:
        errors.append(f"Jumlah transit ({jumlah_transit}) di luar rentang valid (1–10 hub).")
    if volume_hub < 100 or volume_hub > 50000:
        errors.append(f"Volume hub ({volume_hub:,}) di luar rentang kapasitas wajar (100–50.000 paket).")

    jarak_val: float | None = None
    if jarak_str.strip() != "":
        try:
            jarak_val = float(jarak_str)
            if jarak_val <= 0 or jarak_val > 5000:
                errors.append(f"Jarak ({jarak_val} km) mustahil. Masukkan 1–5.000 km atau kosongkan untuk otomatis.")
        except ValueError:
            errors.append("Jarak harus berupa angka atau dikosongkan agar diisi otomatis.")

    return errors, jarak_val


def deteksi_faktor_risiko(row: pd.Series) -> list[str]:
    faktor = []
    if row["volume_hub_hari"] > META["imputation_stats"]["vol_q75"]:
        faktor.append(f"Volume hub tinggi ({int(row['volume_hub_hari']):,} paket/hari)")
    if row["jam_pickup"] >= 17:
        faktor.append(f"Pickup sore/malam pukul {int(row['jam_pickup'])}:00 (mendekati cutoff armada)")
    if row["kecepatan_wajib_km_per_hari"] > 250:
        faktor.append(f"Target jarak harian ketat ({row['kecepatan_wajib_km_per_hari']:.0f} km/hari)")
    if row["jumlah_transit"] >= 3:
        faktor.append(f"Banyak titik transit ({int(row['jumlah_transit'])} hub)")
    if row["cuaca_saat_pickup"] in ("hujan", "hujan_lebat"):
        faktor.append(f"Cuaca kurang mendukung ({row['cuaca_saat_pickup']})")
    return faktor or ["Kondisi operasional relatif normal"]


st.title("📦 Sistem Prediksi Keterlambatan Paket — Hub Bandung")
st.caption("Alat bantu staf layanan pelanggan dan operasional hub saat penjemputan paket.")

with st.sidebar:
    st.header("ℹ️ Ringkasan Model")
    st.write(f"**Model:** {META['model_name']}")
    st.write(f"**Val PR-AUC:** {META['metrics']['validation']['final_pr_auc']}")
    st.write(f"**Test PR-AUC:** {META['metrics']['test']['final_pr_auc']}")
    st.divider()
    st.write(f"**Batas Kabari Pelanggan:** ≥ {META['thresholds']['kabari_pelanggan']:.1%}")
    st.write(f"**Batas Prioritas Sortir:** ≥ {META['thresholds']['prioritaskan_paket']:.1%}")
    st.info(
        "Dasar biaya bisnis:\n"
        "- Terlambat tanpa kabar: **Rp50.000**\n"
        "- Notifikasi keliru: **Rp5.000**"
    )

tab1, tab2 = st.tabs(["🔍 Cek Satu Paket", "📂 Prediksi Massal (CSV)"])

with tab1:
    with st.form("form_prediksi_paket"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tanggal_pickup = st.date_input("Tanggal Pickup", value=date(2026, 5, 12))
            jam_pickup = st.number_input("Jam Pickup (07–22)", min_value=0, max_value=24, value=14, step=1)
            kota_tujuan = st.selectbox("Kota Tujuan", options=KOTA_PILIHAN, index=KOTA_PILIHAN.index("Jakarta"))
            jarak_str = st.text_input(
                "Jarak (km) — kosongkan jika otomatis",
                value="",
                placeholder="Contoh: 150 (atau biarkan kosong)",
            )
        with col2:
            layanan = st.selectbox("Jenis Layanan", options=["EXPRESS", "REG", "HEMAT"], index=1)
            janji_hari = st.number_input("Janji SLA (hari)", min_value=0, max_value=30, value=2, step=1)
            jumlah_transit = st.number_input("Jumlah Hub Transit", min_value=0, max_value=20, value=1, step=1)
            berat_kg = st.number_input("Berat Paket (kg)", min_value=-50.0, max_value=500.0, value=1.5, step=0.1)
        with col3:
            metode_bayar = st.selectbox("Metode Pembayaran", options=["non_COD", "COD"])
            cuaca = st.selectbox("Cuaca Saat Pickup", options=["cerah", "berawan", "hujan", "hujan_lebat"])
            volume_hub = st.number_input("Volume Hub Hari Ini", min_value=-1000, max_value=100000, value=5400, step=100)

        submitted = st.form_submit_button("Prediksi Risiko Keterlambatan", use_container_width=True)

    if submitted:
        daftar_error, jarak_val = validasi_input(
            kota_tujuan, jam_pickup, jarak_str, berat_kg, janji_hari, jumlah_transit, volume_hub
        )
        if daftar_error:
            st.error("🚫 **Masukan Tidak Wajar Terdeteksi — Prediksi Dibatalkan:**")
            for err in daftar_error:
                st.write(f"- {err}")
        else:
            df_in = pd.DataFrame([{
                "tanggal_pickup": str(tanggal_pickup),
                "jam_pickup": int(jam_pickup),
                "kota_tujuan": kota_tujuan,
                "jarak_km": jarak_val if jarak_val is not None else np.nan,
                "jumlah_transit": int(jumlah_transit),
                "layanan": layanan,
                "janji_hari": int(janji_hari),
                "berat_kg": float(berat_kg),
                "metode_bayar": metode_bayar,
                "cuaca_saat_pickup": cuaca,
                "volume_hub_hari": int(volume_hub),
            }])

            cleaned = clean_dataframe(df_in, deduplicate=False)
            featured = apply_imputation_and_features(cleaned, META["imputation_stats"])
            cols = META["features_num"] + META["features_cat"]
            prob = float(MODEL.predict_proba(featured[cols])[0, 1])

            th_kabari = META["thresholds"]["kabari_pelanggan"]
            th_prioritas = META["thresholds"]["prioritaskan_paket"]

            if jarak_val is None:
                st.info(f"ℹ️ Jarak dikosongkan → diisi otomatis dengan median rute **{kota_tujuan}**: **{featured.iloc[0]['jarak_km']:.0f} km**.")

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Probabilitas Terlambat", f"{prob:.1%}")
            with c2:
                if prob >= th_prioritas:
                    st.error("🚨 **REKOMENDASI: PRIORITASKAN SORTIR & KABARI PELANGGAN**")
                elif prob >= th_kabari:
                    st.warning("⚠️ **REKOMENDASI: KABARI PELANGGAN LEBIH AWAL**")
                else:
                    st.success("✅ **REKOMENDASI: AMAN (PROSES REGULER)**")

            st.subheader("Faktor Risiko Terdeteksi:")
            for f in deteksi_faktor_risiko(featured.iloc[0]):
                st.write(f"• {f}")

with tab2:
    st.write("Unggah file CSV data paket saat pickup untuk memprediksi banyak paket sekaligus.")
    uploaded = st.file_uploader("Pilih berkas CSV", type=["csv"])
    if uploaded is not None:
        df_batch = pd.read_csv(uploaded)
        cleaned_b = clean_dataframe(df_batch, deduplicate=False)
        featured_b = apply_imputation_and_features(cleaned_b, META["imputation_stats"])
        cols = META["features_num"] + META["features_cat"]
        probs = MODEL.predict_proba(featured_b[cols])[:, 1]

        th_k = META["thresholds"]["kabari_pelanggan"]
        th_p = META["thresholds"]["prioritaskan_paket"]
        df_batch["probabilitas_terlambat"] = probs.round(4)
        df_batch["rekomendasi"] = [
            "PRIORITASKAN & KABARI" if p >= th_p else ("KABARI PELANGGAN" if p >= th_k else "AMAN")
            for p in probs
        ]
        st.dataframe(df_batch.head(50), use_container_width=True)
        st.download_button(
            "Unduh Hasil Prediksi (CSV)",
            data=df_batch.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_keterlambatan.csv",
            mime="text/csv",
        )
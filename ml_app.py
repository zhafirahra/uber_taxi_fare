# Simpan kode ini sebagai file Python (misalnya, app.py) dan jalankan dengan 'streamlit run app.py'

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math # Import math for haversine function
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor # Import the model

# --- Definisikan Fungsi Rekayasa Fitur (Harus sama dengan saat training) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Jari-jari bumi dalam kilometer
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def get_season(mnth):
  if mnth in [12,1,2]:
    return 'winter'
  elif mnth in [3,4,5]:
    return 'spring'
  elif mnth in [6,7,8]:
    return 'summer'
  elif mnth in [9,10,11]:
    return 'fall'
  return None # Handle unexpected months if any

def hour_category(hour):
    if 7 <= hour <= 9 or 16 <= hour <= 19:
        return 'rush_hour'
    elif 0 <= hour <= 5:
        return 'night'
    else:
        return 'off_peak'

# --- Muat Model dan Preprocessor yang Sudah Disimpan (Pickle) ---
try:
    with open('best_gradient_boosting_model_no_log.pkl', 'rb') as f:
        model_best = pickle.load(f)
    with open('scaler_no_log.pkl', 'rb') as f:
        scaler_best = pickle.load(f)
    with open('ordinal_encoder_no_log.pkl', 'rb') as f:
        ordinal_encoder_best = pickle.load(f)
    with open('one_hot_encoder_no_log.pkl', 'rb') as f:
        ohe_best = pickle.load(f)
    st.success("Model dan preprocessor berhasil dimuat!")
except FileNotFoundError:
    st.error("File model/preprocessor tidak ditemukan. Pastikan Anda sudah menjalankan kode pickling.")
    st.stop() # Hentikan aplikasi jika file tidak ditemukan

# --- Konfigurasi Fitur (Harus sama dengan saat training) ---
# Tentukan kolom berdasarkan jenis (harus sesuai dengan fitur pelatihan model terbaik)
numeric_features_best = ['passenger_count', 'dist', 'pickup_year']
ordinal_features_best = ['pickup_season']
nominal_features_best = ['pickup_month', 'pickup_weekday', 'pickup_hour', 'pickup_hour_category']

# Dapatkan nama-nama kolom yang diharapkan setelah OHE dari encoder yang sudah dilatih
ohe_feature_names = ohe_best.get_feature_names_out(nominal_features_best)
# Gabungkan semua nama kolom yang diharapkan untuk data final
# Ini perlu disesuaikan agar sesuai dengan kolom training yang benar dari model terbaik
# Berdasarkan analisis terakhir, model terbaik menggunakan data TANPA log transform pada dist
# dan outlier fare/dist dihapus. Nama kolom dari OHE dan OrdinalEncoder harus sama.
# Kita perlu mendapatkan nama kolom final dari X_train_final_best yang benar.

# Untuk memastikan urutan kolom sama, cara terbaik adalah mendapatkan daftar kolom dari X_train_final_best
# yang digunakan saat melatih model_best.
# Karena kita tidak memiliki X_train_final_best dari cell yang sama persis di sini,
# kita bisa merekonstruksi nama kolom atau asumsikan urutannya konsisten.
# Asumsi: Urutan kolom adalah numeric_features, ordinal_features, lalu one-hot encoded features.

# Rekonstruksi nama kolom OHE:
# Ini adalah nama-nama kolom yang dihasilkan oleh ohe_best.get_feature_names_out
# Kita bisa menggunakannya secara langsung.
# Jika ada masalah urutan kolom, kita perlu memastikan urutan saat membuat DataFrame input_data_final.

# --- Antarmuka Pengguna (UI) Streamlit ---
st.title("Prediksi Tarif Taksi Uber")
st.write("Masukkan detail perjalanan untuk memprediksi tarif.")

# Input dari pengguna
passenger_count = st.number_input("Jumlah Penumpang", min_value=1, max_value=6, value=1, step=1)
pickup_date = st.date_input("Tanggal Penjemputan")
pickup_time = st.time_input("Waktu Penjemputan")
pickup_lat = st.number_input("Latitude Penjemputan", value=40.738354, format="%.6f")
pickup_lon = st.number_input("Longitude Penjemputan", value=-73.999817, format="%.6f")
dropoff_lat = st.number_input("Latitude Tujuan", value=40.723217, format="%.6f")
dropoff_lon = st.number_input("Longitude Tujuan", value=-73.999512, format="%.6f")


# Tombol Prediksi
if st.button("Prediksi Tarif"):
    # --- Persiapan Data Input Pengguna untuk Prediksi ---
    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'passenger_count': [passenger_count],
        'pickup_datetime': [pd.to_datetime(f"{pickup_date} {pickup_time}")],
        'pickup_longitude': [pickup_lon],
        'pickup_latitude': [pickup_lat],
        'dropoff_longitude': [dropoff_lon],
        'dropoff_latitude': [dropoff_lat],
    })

    # Terapkan rekayasa fitur yang sama
    input_data['pickup_year'] = input_data['pickup_datetime'].dt.year
    input_data['pickup_month'] = input_data['pickup_datetime'].dt.month
    input_data['pickup_weekday'] = input_data['pickup_datetime'].dt.weekday
    input_data['pickup_hour'] = input_data['pickup_datetime'].dt.hour
    input_data['pickup_season'] = input_data['pickup_month'].apply(get_season)
    input_data['pickup_hour_category'] = input_data['pickup_hour'].apply(hour_category)

    # Hitung jarak menggunakan fungsi haversine
    input_data['dist'] = input_data.apply(
        lambda row: haversine(
            row['pickup_latitude'], row['pickup_longitude'],
            row['dropoff_latitude'], row['dropoff_longitude']
        ),
        axis=1
    )

    # Hapus kolom yang tidak digunakan model
    input_data = input_data.drop(columns = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "pickup_datetime"])

    # TIDAK ADA Log transform 'dist' pada konfigurasi model terbaik ini


    # Terapkan Preprocessing menggunakan encoder dan scaler yang sudah dilatih
    try:
        input_data_ord = pd.DataFrame(
            ordinal_encoder_best.transform(input_data[ordinal_features_best]),
            columns=ordinal_features_best,
            index=input_data.index
        )

        input_data_cat = pd.DataFrame(
            ohe_best.transform(input_data[nominal_features_best]),
            columns=ohe_feature_names, # Gunakan nama kolom yang diharapkan dari OHE
            index=input_data.index
        )

        input_data_num = pd.DataFrame(
            scaler_best.transform(input_data[numeric_features_best]),
            columns=numeric_features_best,
            index=input_data.index
        )

        # Gabungkan semua fitur
        # Pastikan urutan kolom sesuai dengan X_train_final_best yang benar
        # Berdasarkan cell Gradient Boosting (Drop Outlier Fare & Dist), urutan kolom adalah numerik, ordinal, lalu one-hot encoded
        input_data_final = pd.concat([input_data_num, input_data_ord, input_data_cat], axis=1)


        # --- Lakukan Prediksi ---
        predicted_fare = model_best.predict(input_data_final)

        # --- Tampilkan Hasil Prediksi ---
        st.subheader("Hasil Prediksi Tarif")
        st.write(f"Perkiraan Tarif Taksi Uber adalah: **${predicted_fare[0]:.2f}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input atau melakukan prediksi: {e}")
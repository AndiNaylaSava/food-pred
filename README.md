# prediksi-cadangan-pangan-pemerintah-daerah-cppd-kabupaten-kota-di-sulawesi-tengah
# Prediksi Cadangan Pangan Pemerintah Daerah (CPPD) Kabupaten/Kota di Sulawesi Tengah

Proyek ini dibuat untuk menganalisis dan memprediksi kebutuhan Cadangan Pangan Pemerintah Daerah (CPPD) pada Kabupaten/Kota di Provinsi Sulawesi Tengah menggunakan metode Machine Learning.

Data yang digunakan terdiri dari kepadatan penduduk, jumlah bencana, dan volume CPPD. Analisis dilakukan untuk melihat hubungan antar variabel serta membandingkan performa beberapa model regresi dalam melakukan prediksi kebutuhan cadangan pangan.

## Dataset

Dataset berisi data Kabupaten/Kota di Sulawesi Tengah dengan variabel:

* Kepadatan Jiwa per km²
* Jumlah Bencana
* CPPD (Ton)

## Metode

Pada penelitian ini digunakan dua model Machine Learning, yaitu:

* Support Vector Regression (SVR)
* Random Forest Regressor

Evaluasi model dilakukan menggunakan Leave-One-Out Cross Validation (LOO-CV) karena jumlah data yang relatif kecil.

## Analisis yang Dilakukan

* Statistik deskriptif
* Uji normalitas Shapiro-Wilk
* Korelasi Pearson
* Perbandingan performa model
* Prediksi kebutuhan CPPD
* Klasifikasi tingkat kebutuhan cadangan pangan

## Output

Program menghasilkan:

* Nilai MAE, RMSE, dan R²
* Model terbaik berdasarkan hasil evaluasi
* Estimasi kebutuhan CPPD setiap Kabupaten/Kota
* Kategori kebutuhan (Rendah, Sedang, Tinggi)
* Visualisasi ranking kebutuhan cadangan pangan

## Teknologi

* Python
* Pandas
* NumPy
* SciPy
* Scikit-Learn
* Matplotlib

Proyek ini disusun sebagai tugas mata kuliah Statistik dan Probabilitas.

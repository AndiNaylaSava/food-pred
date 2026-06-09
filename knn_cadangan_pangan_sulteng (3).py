import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# ─────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────
df_raw = pd.read_excel("dataset_cadangan_pangan_sulteng.xlsx")
 
# Gunakan rata-rata CPPD per kabupaten (hapus duplikasi bulan)
df = df_raw.groupby("Kabupaten/Kota").agg(
    Kepadatan_Jiwa_per_km2=("Kepadatan_Jiwa_per_km2", "first"),
    Jumlah_Bencana=("Jumlah_Bencana", "first"),
    CPPD=("CPPD (Ton)", "mean")
).reset_index()
 
print("DATASET CADANGAN PANGAN SULAWESI TENGAH")
print(f"Jumlah Kab/Kota : {len(df)}")
print(f"Rata-rata CPPD  : {df['CPPD'].mean():.2f} ton")
print()
print(df[["Kabupaten/Kota", "Kepadatan_Jiwa_per_km2", "Jumlah_Bencana", "CPPD"]].to_string(index=False))
print()
 
# ─────────────────────────────────────────────
# Statistik Deskriptif (NumPy + SciPy)
# ─────────────────────────────────────────────
cppd = df["CPPD"].values
 
print("STATISTIK DESKRIPTIF CPPD")
print(f"Mean     : {np.mean(cppd):.4f} ton")
print(f"Median   : {np.median(cppd):.4f} ton")
print(f"Std Dev  : {np.std(cppd):.4f} ton")
print(f"Min      : {np.min(cppd):.4f} ton")
print(f"Max      : {np.max(cppd):.4f} ton")
print(f"Skewness : {stats.skew(cppd):.4f}")
print(f"Kurtosis : {stats.kurtosis(cppd):.4f}")
 
stat_sw, p_sw = stats.shapiro(cppd)
print(f"Shapiro-Wilk : stat={stat_sw:.4f}, p={p_sw:.4f}", end="  →  ")
print("Normal" if p_sw > 0.05 else "Tidak Normal")
 
print()
print("KORELASI PEARSON FITUR vs CPPD")
for fitur in ["Kepadatan_Jiwa_per_km2", "Jumlah_Bencana"]:
    r, p = stats.pearsonr(df[fitur], df["CPPD"])
    print(f"  {fitur:<25} r={r:+.4f}  p={p:.4f}", end="  →  ")
    print("Signifikan" if p < 0.05 else "Tidak Signifikan")
print()
 
# ─────────────────────────────────────────────
# Fitur dan Target
# ─────────────────────────────────────────────
FITUR  = ["Kepadatan_Jiwa_per_km2", "Jumlah_Bencana"]
TARGET = "CPPD"
 
X = df[FITUR].values
y = df[TARGET].values
 
# ─────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# ─────────────────────────────────────────────
# Evaluasi dengan Leave-One-Out CV
# (karena n=13 terlalu kecil untuk train/test split)
# ─────────────────────────────────────────────
from sklearn.model_selection import KFold
 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
 
models = {
    "SVR":           SVR(kernel="rbf", C=100, gamma="scale"),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}
 
print("PERBANDINGAN MODEL (5-Fold Cross Validation)")
print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8}")
print("-" * 40)
 
best_model_name = None
best_rmse       = 999
 
for nama, model in models.items():
    X_input = X_scaled if nama == "SVR" else X
 
    mae_cv  = -cross_val_score(model, X_input, y, cv=kf,
                                scoring="neg_mean_absolute_error").mean()
    mse_cv  = -cross_val_score(model, X_input, y, cv=kf,
                                scoring="neg_mean_squared_error").mean()
    rmse_cv = np.sqrt(mse_cv)
 
    print(f"{nama:<20} {mae_cv:>8.4f} {rmse_cv:>8.4f}")
 
    if rmse_cv < best_rmse:
        best_rmse       = rmse_cv
        best_model_name = nama
 
print()
print(f"Model Terbaik : {best_model_name}")
print(f"RMSE CV       : {best_rmse:.4f} ton")
print()
 
# ─────────────────────────────────────────────
# Training model terbaik dengan seluruh data
# lalu prediksi semua kabupaten
# ─────────────────────────────────────────────
best_model = models[best_model_name]
X_input    = X_scaled if best_model_name == "SVR" else X
best_model.fit(X_input, y)
y_pred = best_model.predict(X_input)
 
mae  = mean_absolute_error(y, y_pred)
mse  = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y, y_pred)
 
print(f"MATRIKS AKURASI — {best_model_name}")
print(f"MAE      : {mae:.4f} ton   ← rata-rata selisih prediksi vs aktual")
print(f"MSE      : {mse:.4f}       ← rata-rata kuadrat selisih")
print(f"RMSE     : {rmse:.4f} ton  ← akar MSE, satuan sama dengan CPPD")
print(f"R² Score : {r2:.4f}        ← {r2*100:.2f}% variansi data dijelaskan model")
print()
 
# Feature Importance (Random Forest)
if best_model_name == "Random Forest":
    print("TINGKAT PENGARUH FITUR")
    importance = sorted(
        zip(FITUR, best_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for fitur, nilai in importance:
        print(f"  {fitur:<25} {nilai:.4f}")
    print()
 
# ─────────────────────────────────────────────
# Grafik Aktual vs Prediksi per Kabupaten/Kota
# ─────────────────────────────────────────────
nama_kab = df["Kabupaten/Kota"].tolist()
x_pos    = np.arange(len(nama_kab))
w        = 0.38
 
plt.figure(figsize=(13, 6))
plt.bar(x_pos - w/2, y,      w, label="Aktual",
        color="#1f77b4", alpha=0.85)
plt.bar(x_pos + w/2, y_pred, w, label=f"Prediksi {best_model_name}",
        color="#ff7f0e", alpha=0.85)
 
plt.xticks(x_pos, nama_kab, rotation=40, ha="right", fontsize=8)
plt.ylabel("CPPD (ton)")
plt.title(
    f"{best_model_name} — Prediksi Cadangan Pangan Sulawesi Tengah\n"
    f"R² = {r2:.4f}  |  MAE = {mae:.4f}  |  RMSE = {rmse:.4f} ton"
)
plt.legend()
plt.tight_layout()
plt.show()
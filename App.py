# streamlit_status_prediksi.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# -------------------------------
# 1. Load & Preprocessing Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/data.csv", sep=';')
    if "Status" not in df.columns:
        st.error("❌ Kolom 'Status' tidak ditemukan di data.csv.")
        st.stop()
    df = df[df["Status"].notna()]
    return df

df = load_data()

# -------------------------------
# 2. Sidebar Input Data
# -------------------------------
st.sidebar.title("🎓 Masukkan Data Mahasiswa")

def user_input_features():
    age = st.sidebar.number_input("Umur saat masuk", min_value=17, max_value=60, value=21)
    admission_grade = st.sidebar.number_input("Admission Grade", min_value=90.0, max_value=200.0, value=150.0)
    prev_grade = st.sidebar.number_input("Previous Qualification Grade", min_value=80.0, max_value=200.0, value=140.0)

    scholarship = st.sidebar.selectbox("Penerima Beasiswa?", options=["Tidak", "Iya"])
    debtor = st.sidebar.selectbox("Memiliki Tanggungan Utang?", options=["Tidak", "Iya"])
    tuition_paid = st.sidebar.selectbox("Biaya Terbayar?", options=["Tidak", "Iya"])

    units1_enrolled = st.sidebar.number_input("Unit Semester 1 Diambil", min_value=0, max_value=20, value=10)
    units1_approved = st.sidebar.number_input("Unit Semester 1 Lulus", min_value=0, max_value=20, value=8)
    eval1 = st.sidebar.number_input("Evaluasi Semester 1", min_value=0, max_value=50, value=30)
    grade1 = st.sidebar.number_input("Nilai Semester 1", min_value=0.0, max_value=20.0, value=14.0)

    units2_enrolled = st.sidebar.number_input("Unit Semester 2 Diambil", min_value=0, max_value=20, value=10)
    units2_approved = st.sidebar.number_input("Unit Semester 2 Lulus", min_value=0, max_value=20, value=8)
    eval2 = st.sidebar.number_input("Evaluasi Semester 2", min_value=0, max_value=50, value=30)
    grade2 = st.sidebar.number_input("Nilai Semester 2", min_value=0.0, max_value=20.0, value=14.0)

    data = {
        "Age_at_enrollment": age,
        "Admission_grade": admission_grade,
        "Previous_qualification_grade": prev_grade,
        "Scholarship_holder": 1 if scholarship == "Iya" else 0,
        "Debtor": 1 if debtor == "Iya" else 0,
        "Tuition_fees_up_to_date": 1 if tuition_paid == "Iya" else 0,
        "Curricular_units_1st_sem_enrolled": units1_enrolled,
        "Curricular_units_1st_sem_approved": units1_approved,
        "Curricular_units_1st_sem_evaluations": eval1,
        "Curricular_units_1st_sem_grade": grade1,
        "Curricular_units_2nd_sem_enrolled": units2_enrolled,
        "Curricular_units_2nd_sem_approved": units2_approved,
        "Curricular_units_2nd_sem_evaluations": eval2,
        "Curricular_units_2nd_sem_grade": grade2
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# -------------------------------
# 3. Model Training & Prediction
# -------------------------------
st.title("🎯 Prediksi Status Mahasiswa")
st.write("Prediksi mahasiswa akan **Dropout**, **Enrolled**, atau **Graduate** berdasarkan fitur akademik dan sosial.")

features = [
    "Age_at_enrollment", "Admission_grade", "Previous_qualification_grade",
    "Scholarship_holder", "Debtor", "Tuition_fees_up_to_date",
    "Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_enrolled", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_grade"
]

X = df[features]
y = df["Status"]

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# -------------------------------
# 4. Output Prediksi
# -------------------------------
st.subheader("🖌️ Hasil Prediksi")
status_map = {
    "Graduate": "🎓 Lulus",
    "Dropout": "⚠️ Dropout",
    "Enrolled": "📚 Masih Terdaftar"
}
label = prediction[0]
st.write(f"**Status Mahasiswa yang Diprediksi:** {status_map.get(label, label)}")

st.subheader("🔍 Probabilitas Kelas")
proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
st.dataframe(proba_df.T.rename(columns={0: "Probabilitas"}))

# -------------------------------
# 5. Visualisasi Tambahan: Analisis Multivariat
# -------------------------------
st.subheader("📈 Analisis Multivariat")

# Drop NA untuk visualisasi
df_clean = df.dropna(subset=["Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade", "Status"])

# Boxplot Nilai Semester
fig1, ax1 = plt.subplots()
sns.boxplot(data=df_clean, x="Status", y="Curricular_units_1st_sem_grade", ax=ax1)
ax1.set_title("Distribusi Nilai Semester 1")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.boxplot(data=df_clean, x="Status", y="Curricular_units_2nd_sem_grade", ax=ax2)
ax2.set_title("Distribusi Nilai Semester 2")
st.pyplot(fig2)

# Barplot rata-rata unit lulus
avg_units = df_clean.groupby("Status")[["Curricular_units_1st_sem_approved", "Curricular_units_2nd_sem_approved"]].mean().reset_index()
fig3, ax3 = plt.subplots()
avg_units.plot(kind='bar', x='Status', ax=ax3)
ax3.set_ylabel("Rata-rata Unit Lulus")
ax3.set_title("Unit Lulus Semester 1 & 2")
st.pyplot(fig3)

# Stacked bar tuition
st.markdown("#### 💰 Distribusi Status Berdasarkan Pembayaran Biaya Kuliah")
fee_status = pd.crosstab(df_clean["Tuition_fees_up_to_date"], df_clean["Status"], normalize="index")
fig4, ax4 = plt.subplots()
fee_status.plot(kind="bar", stacked=True, ax=ax4, colormap="Pastel1")
ax4.set_title("Proporsi Status Berdasarkan Pembayaran")
ax4.set_ylabel("Proporsi")
ax4.legend(title="Status")
st.pyplot(fig4)

# Heatmap korelasi
st.markdown("#### 🧠 Korelasi Fitur dengan Status")
df_corr = df_clean.copy()
df_corr["Status"] = df_corr["Status"].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2})
corr = df_corr.corr(numeric_only=True)
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr[["Status"]].sort_values("Status", ascending=False), annot=True, cmap="coolwarm", ax=ax5)
ax5.set_title("Korelasi Fitur dengan Status Mahasiswa")
st.pyplot(fig5)

# -------------------------------
# 6. Eksplorasi Dataset
# -------------------------------
with st.expander("📊 Lihat Cuplikan Dataset"):
    st.dataframe(df.sample(10))

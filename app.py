


import os
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf_path = os.path.join(BASE_DIR, "classification_model.pkl")
    reg_path = os.path.join(BASE_DIR, "regression_model.pkl")
    with open(clf_path, "rb") as f:
        clf_model = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model


clf_model, reg_model = load_models()

st.title("🎓 Student Placement & Salary Predictor")
st.markdown(
    "Aplikasi ini memprediksi **status penempatan kerja** dan "
    "**estimasi gaji** mahasiswa berdasarkan profil akademik & keterampilan."
)
st.divider()

with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.info(
        "Model:\n"
        "- Klasifikasi: Random Forest\n"
        "- Regresi: Gradient Boosting\n\n"
        "Dataset: 5.000 data mahasiswa (Dataset B)"
    )
    st.markdown("---")

with st.form("prediction_form"):
    st.subheader("📋 Masukkan Data Mahasiswa")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data Akademik**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc = st.slider("SSC Percentage (%)", 50, 95, 70)
        hsc = st.slider("HSC Percentage (%)", 50, 95, 70)
        degree = st.slider("Degree Percentage (%)", 50, 95, 70)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        attendance = st.slider("Attendance (%)", 50, 100, 80)
        backlogs = st.number_input("Jumlah Backlogs", min_value=0, max_value=20, value=0)

    with col2:
        st.markdown("**Ujian & Kemampuan**")
        entrance_score = st.slider("Entrance Exam Score", 0, 100, 60)
        technical_skill = st.slider("Technical Skill Score (0-100)", 0, 100, 60)
        soft_skill = st.slider("Soft Skill Score (0-100)", 0, 100, 60)

    with col3:
        st.markdown("**Pengalaman & Sertifikasi**")
        internship = st.number_input("Jumlah Internship", min_value=0, max_value=10, value=1)
        live_projects = st.number_input("Live Projects", min_value=0, max_value=20, value=2)
        work_exp = st.number_input("Work Experience (bulan)", min_value=0, max_value=60, value=6)
        certifications = st.number_input("Jumlah Sertifikasi", min_value=0, max_value=20, value=2)
        extracurricular = st.selectbox("Ekstrakurikuler", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "gender":                    gender,
        "ssc_percentage":            ssc,
        "hsc_percentage":            hsc,
        "degree_percentage":         degree,
        "cgpa":                      cgpa,
        "entrance_exam_score":       entrance_score,
        "technical_skill_score":     technical_skill,
        "soft_skill_score":          soft_skill,
        "internship_count":          internship,
        "live_projects":             live_projects,
        "work_experience_months":    work_exp,
        "certifications":            certifications,
        "attendance_percentage":     attendance,
        "backlogs":                  backlogs,
        "extracurricular_activities": extracurricular,
    }])

    placement_pred = clf_model.predict(input_data)[0]
    placement_prob = clf_model.predict_proba(input_data)[0][1]
    salary_pred    = max(0.0, reg_model.predict(input_data)[0])

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    col_a, col_b = st.columns(2)

    with col_a:
        if placement_pred == 1:
            st.success("✅ **Status: PLACED** – Mahasiswa ini diprediksi berhasil mendapatkan penempatan kerja.")
        else:
            st.warning("⚠️ **Status: NOT PLACED** – Mahasiswa ini diprediksi belum berhasil mendapatkan penempatan kerja.")
        st.metric("Probabilitas Placed", f"{placement_prob:.1%}")

    with col_b:
        if placement_pred == 1:
            st.info(f"💰 **Estimasi Gaji: {salary_pred:.2f} LPA**")
            st.metric("Salary Package (LPA)", f"{salary_pred:.2f}")
        else:
            st.info("💰 Estimasi gaji hanya tersedia untuk mahasiswa yang diprediksi **Placed**.")

    # Detail input yang dimasukkan
    with st.expander("📄 Lihat Detail Input"):
        st.dataframe(input_data.T.rename(columns={0: "Nilai"}), use_container_width=True)

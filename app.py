import os
import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Student Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    clf_path = os.path.join(BASE_DIR, "classification_model.pkl")
    reg_path = os.path.join(BASE_DIR, "regression_model.pkl")
    
    with open(clf_path, "rb") as f:
        clf_model = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model

try:
    clf_model, reg_model = load_models()
except FileNotFoundError:
    st.error("⚠️ Model tidak ditemukan di folder 'artifacts/'. Pastikan Anda sudah menjalankan pipeline.py!")
    st.stop()

# Navigasi Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/a/ad/Logo_Binus_University.png", width=180)
st.sidebar.markdown("---")
st.sidebar.title("🧭 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "🔮 Prediksi Karir", "📈 Wawasan Data"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Sistem Monolithic**\n\n"
    "Arsitektur terintegrasi untuk analitik data sains dan prediksi *machine learning*."
)
st.sidebar.caption("UTS Model Deployment © 2026")

# Beranda
if menu == "🏠 Beranda":
    st.title("🎓 Student Placement & Salary Analytics")
    st.markdown("Selamat datang di portal prediksi analitik karir mahasiswa berbasis **Machine Learning**.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Dataset Pelatihan", value="5,000", delta="Dataset B")
    col2.metric(label="Algoritma Klasifikasi", value="Random Forest", delta="F1-Score Optimasi", delta_color="normal")
    col3.metric(label="Algoritma Regresi", value="Random Forest", delta="MAE Optimasi", delta_color="normal")

    st.markdown("### 📌 Petunjuk Penggunaan")
    st.info(
        """
        1. Buka menu **🔮 Prediksi Karir** di sidebar sebelah kiri.
        2. Masukkan parameter akademik, kemampuan teknis, dan pengalaman Anda.
        3. Sistem akan mengalkulasi probabilitas Anda diterima bekerja beserta estimasi *Salary Package* (LPA).
        4. Buka menu **📈 Wawasan Data** untuk melihat tren umum dari mahasiswa.
        """
    )
    
    
    fig_hero = px.scatter(x=[1, 2, 3], y=[1, 3, 2], title="Membangun Karir Berbasis Data", template="plotly_white")
    fig_hero.update_traces(marker=dict(size=20, color="#1f77b4", symbol="diamond"))
    fig_hero.update_layout(height=300, xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(fig_hero, use_container_width=True)

# Prediksi
elif menu == "🔮 Prediksi Karir":
    st.title("🔮 Kalkulator Prediksi Penempatan")
    st.markdown("Sesuaikan parameter di bawah ini untuk melihat simulasi profil Anda.")
    
    with st.form("prediction_form"):
        
        tab1, tab2, tab3 = st.tabs(["📚 Akademik", "💡 Skill & Ujian", "💼 Pengalaman"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                ssc = st.slider("SSC % (Sekolah Menengah)", 0, 100, 75)
                hsc = st.slider("HSC % (SMA)", 0, 100, 70)
            with col2:
                degree = st.slider("Degree % (Kuliah)", 0, 100, 72)
                cgpa = st.number_input("IPK (CGPA)", 0.0, 10.0, 7.5, 0.01)
                attendance = st.slider("Kehadiran (%)", 0, 100, 85)
                backlogs = st.number_input("Jumlah Mata Kuliah Mengulang", 0, 20, 0)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                entrance_score = st.slider("Skor Ujian Masuk", 0, 100, 70)
                tech_skill = st.slider("Technical Skill Score", 0, 100, 80)
            with col2:
                soft_skill = st.slider("Soft Skill Score", 0, 100, 75)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                internship = st.number_input("Jumlah Magang", 0, 10, 1)
                projects = st.number_input("Live Projects", 0, 20, 2)
                work_exp = st.number_input("Pengalaman Kerja (Bulan)", 0, 100, 6)
            with col2:
                certs = st.number_input("Jumlah Sertifikasi", 0, 20, 3)
                extra = st.selectbox("Aktif Ekstrakurikuler?", ["Yes", "No"])

        st.markdown("---")
        submitted = st.form_submit_button("🚀 Proses Analisis", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "gender": gender, "ssc_percentage": ssc, "hsc_percentage": hsc,
            "degree_percentage": degree, "cgpa": cgpa, "entrance_exam_score": entrance_score,
            "technical_skill_score": tech_skill, "soft_skill_score": soft_skill,
            "internship_count": internship, "live_projects": projects,
            "work_experience_months": work_exp, "certifications": certs,
            "attendance_percentage": attendance, "backlogs": backlogs,
            "extracurricular_activities": extra
        }])

        placement = clf_model.predict(input_data)[0]
        prob = clf_model.predict_proba(input_data)[0][1]
        salary = reg_model.predict(input_data)[0] if placement == 1 else 0.0

        st.divider()
        st.subheader("📊 Hasil Keputusan Model")
        
        res_col1, res_col2 = st.columns([1, 1.5])

        with res_col1:
            if placement == 1:
                st.success("### 🎉 STATUS: PLACED")
                st.metric("Estimasi Gaji (LPA)", f"₹ {salary:.2f}")
            else:
                st.error("### 🛑 STATUS: NOT PLACED")
                st.metric("Estimasi Gaji", "Tidak Memenuhi Syarat")
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                title = {'text': "Confidence Level (%)"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': "#2ecc71" if placement == 1 else "#e74c3c"},
                         'steps': [{'range': [0, 50], 'color': "#ffeaa7"},
                                   {'range': [50, 100], 'color': "#dff9fb"}]}
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.write("#### 🕸️ Pemetaan Kompetensi Individu")
            categories = ['Technical', 'Soft Skills', 'Akademik (CGPA x10)', 'Ujian Masuk', 'Kehadiran']
            values = [tech_skill, soft_skill, cgpa * 10, entrance_score, attendance]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=categories, fill='toself', 
                name='Profil Anda', marker_color="#0984e3"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, height=350, margin=dict(l=50, r=50, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# Wawasan DATA
elif menu == "📈 Wawasan Data":
    st.title("📈 Wawasan & Distribusi Data")
    st.markdown("Visualisasi interaktif ini mensimulasikan metrik penting yang memengaruhi penempatan kerja.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dampak Sertifikasi terhadap Gaji")
       
        np.random.seed(42)
        dummy_certs = np.random.randint(0, 10, 100)
        dummy_salary = dummy_certs * 0.5 + np.random.normal(5, 2, 100)
        df_viz1 = pd.DataFrame({"Sertifikasi": dummy_certs, "Gaji (LPA)": dummy_salary})
        
        fig1 = px.box(df_viz1, x="Sertifikasi", y="Gaji (LPA)", color="Sertifikasi", 
                      color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Korelasi Skill Teknis vs Soft Skill")
        dummy_tech = np.random.normal(70, 15, 100)
        dummy_soft = np.random.normal(65, 15, 100)
        dummy_status = np.where(dummy_tech + dummy_soft > 140, "Placed", "Not Placed")
        df_viz2 = pd.DataFrame({"Tech Skill": dummy_tech, "Soft Skill": dummy_soft, "Status": dummy_status})
        
        fig2 = px.scatter(df_viz2, x="Tech Skill", y="Soft Skill", color="Status", 
                          color_discrete_map={"Placed": "#2ecc71", "Not Placed": "#e74c3c"})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.info("💡 **Insight:** Mahasiswa dengan keseimbangan yang baik antara skor teknis dan keterampilan komunikasi (Soft Skills) secara historis memiliki tingkat penempatan kerja yang jauh lebih tinggi.")

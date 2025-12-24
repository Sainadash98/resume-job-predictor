import streamlit as st
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="centered"
)

# ================= LOAD MODELS =================
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ================= CUSTOM UI =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141E30, #243B55);
}

.main {
    background-color: #0e1117;
    padding: 30px;
    border-radius: 16px;
}

h1 {
    text-align: center;
    color: #ffffff;
}

textarea {
    border-radius: 10px;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ================= UI CONTENT =================
st.title("📄 AI Resume Classification System")
st.write("🔍 *Paste your resume text and let AI predict your job category*")

st.write("✔️ TF-IDF Model Loaded:", hasattr(tfidf, "idf_"))

resume_text = st.text_area(
    "📌 Paste Resume Text Here",
    height=300,
    placeholder="Paste your complete resume content here..."
)

if st.button("🚀 Predict Category"):
    if resume_text.strip() == "":
        st.warning("⚠️ Please paste resume text")
    else:
        text_vector = tfidf.transform([resume_text])
        prediction = model.predict(text_vector)
        category = le.inverse_transform(prediction)[0]
        st.success(f"✅ Predicted Job Category: **{category}**")

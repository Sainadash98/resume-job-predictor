import streamlit as st
import pickle
import pdfplumber
import docx

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="centered"
)

# ---------------- LOAD MODELS ----------------
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ---------------- UI HEADER ----------------
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(to right, #141E30, #243B55);
            color: white;
        }
        textarea {
            background-color: #f5f5f5 !important;
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 AI Resume Job Predictor")
st.write("Upload your resume and predict the job category instantly.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

resume_text = ""

# ---------------- TEXT EXTRACTION ----------------
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                resume_text += page.extract_text()

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            resume_text += para.text

    st.text_area("Extracted Resume Text", resume_text, height=250)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Job Role"):
    if resume_text.strip() == "":
        st.warning("Please upload a valid resume")
    else:
        text_vector = tfidf.transform([resume_text])
        prediction = model.predict(text_vector)
        category = le.inverse_transform(prediction)[0]

        st.success(f"✅ Predicted Job Category: **{category}**")

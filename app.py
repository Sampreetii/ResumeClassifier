import streamlit as st
import pickle
import re
import string
import PyPDF2
import io

# Load model, vectorizer, and encoder
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('clf.pkl', 'rb') as f:
    model = pickle.load(f)

def cleanResume(txt):
    txt = txt.lower()
    txt = re.sub(r'http\S+|www\S+', ' ', txt)
    txt = re.sub(r'\S+@\S+', ' ', txt)
    txt = re.sub(r'@\w+', ' ', txt)
    txt = re.sub(r'#\w+', ' ', txt)
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = re.sub(r'[^\x00-\x7F]+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def predict_category(resume_text):
    cleaned = cleanResume(resume_text)
    vectorized = tfidf.transform([cleaned]).toarray()
    pred = model.predict(vectorized)
    return le.inverse_transform(pred)[0]

# Streamlit UI
st.set_page_config(
    page_title="Resume Category Classifier",
    page_icon="resumeicon.jpg",
    layout="centered"
)
st.title("Resume Category Classifier")
st.write("Upload a PDF resume or paste text below to predict the job category.")

# File upload option
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

# Text input option
st.write("--- OR ---")
resume_input = st.text_area("Paste Resume Text Here", height=300)

# Prediction logic
if st.button("Predict"):
    if uploaded_file is not None:
        # Process PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.write("**Extracted text from PDF:**")
            st.text_area("PDF Content", pdf_text, height=200, disabled=True)
            category = predict_category(pdf_text)
            st.success(f"Predicted Category: **{category}**")
    elif resume_input.strip():
        # Process text input
        category = predict_category(resume_input)
        st.success(f"Predicted Category: **{category}**")
    else:
        st.warning("Please upload a PDF file or paste resume text to predict.") 

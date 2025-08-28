import streamlit as st
import pickle, re, string, pandas as pd
import pdfplumber

# ===== Load model, vectorizer, encoder =====
tfidf = pickle.load(open('tfidf.pkl','rb'))
le    = pickle.load(open('encoder.pkl','rb'))
model = pickle.load(open('clf.pkl','rb'))

def cleanResume(txt):
    txt = txt.lower()
    txt = re.sub(r'http\S+|www\S+|\S+@\S+|@\w+|#\w+|<.*?>',' ',txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = re.sub(r'[^\x00-\x7F]+',' ',txt)
    return re.sub(r'\s+',' ',txt).strip()

def extract_text_from_pdf(uploaded_file):
    """Extract text using pdfplumber only."""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
    return text

def predict_category(resume_text):
    cleaned = cleanResume(resume_text)
    vectorized = tfidf.transform([cleaned]).toarray()
    pred = model.predict(vectorized)
    return le.inverse_transform(pred)[0]

# ===== Streamlit UI =====
st.set_page_config(page_title="Resume Category Classifier", page_icon="📄")

st.title("📄 Resume Category Classifier")
st.write("Upload multiple PDF resumes to predict job categories.")

uploaded_files = st.file_uploader("Choose PDF resumes", type=["pdf"], accept_multiple_files=True)

if st.button("Predict Categories"):
    if uploaded_files:
        results = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if not text.strip():
                st.warning(f"No text found in {file.name}. This PDF may be scanned or image-only.")
                continue
            category = predict_category(text)
            results.append({"Filename": file.name, "Predicted Category": category})

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", csv, "resume_predictions.csv", "text/csv")
    else:
        st.warning("Please upload at least one PDF file.")

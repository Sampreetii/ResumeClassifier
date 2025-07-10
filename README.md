# Project Explanation: Resume Category Classifier

## Why I Built This

Recruiters often deal with hundreds or thousands of resumes for different job roles. Manually screening each resume to understand the candidate’s domain (e.g., Data Science, Accounting, Aviation) is slow, inefficient, and error-prone.

I wanted to automate this first layer of screening by building a Resume Category Classifier that can:
- Understand raw resume content
- Predict the most likely job domain based on skills, tools, and job history
- Help recruiters or job portals in routing resumes more intelligently

This project also allowed me to apply practical NLP and machine learning skills to solve a real-world HR tech problem.

---

## How It Works

### 1. Text Preprocessing
- Resumes are cleaned using a custom `cleanResume()` function.
- This removes emails, URLs, special characters, and normalizes whitespace.
- Keywords like known skills (Python, SAP, Tableau, etc.) are emphasized to improve model learning.

### 2. Feature Extraction
- Cleaned resumes are converted into numerical features using **TF-IDF Vectorization**.
- This captures important terms and term frequency across resumes.

### 3. Model Training
- A **Logistic Regression** model (or SVM) is trained to classify resumes into predefined categories (e.g., Software Engineer, Accountant, Aviation, etc.).
- Class imbalance is handled using `RandomOverSampler` from `imbalanced-learn`.

### 4. Evaluation
- The model is evaluated using cross-validation and tested on unseen resumes.
- Performance is measured using accuracy and classification report.

### 5. Prediction
- New resume input is cleaned, vectorized, and passed to the trained model.
- Output is the predicted job category label.

---

## Real-World Applications

- Resume triaging in ATS (Applicant Tracking Systems)
- Candidate-job matching on job portals
- Internal talent classification for HR analytics
- Building smarter interview bots and career recommendation tools

---

## What I Learned

- Practical end-to-end ML pipeline: cleaning, feature engineering, model training
- Handling class imbalance and overfitting
- The importance of preprocessing in NLP problems
- Real-world use of classification in recruitment tech


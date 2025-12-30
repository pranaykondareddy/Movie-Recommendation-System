import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import re

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

def extract_contact_email(text):
    email = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return email[0] if email else "Not found"

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_rank_label(score):
    if score >= 80:
        return "Excellent ✅"
    elif score >= 50:
        return "Good ⚡"
    else:
        return "Needs Improvement ❌"

def check_resume_length(text):
    word_count = len(text.split())
    if word_count < 300:
        return "Too short ❌"
    elif word_count > 1000:
        return "Too long ⚠️"
    else:
        return "Good length ✅"

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    scores = cosine_similarity([job_vector], resume_vectors).flatten()
    return scores

# Streamlit UI
st.title("ML Resume Ranking System")

st.header("Job Description")
job_description = st.text_area("Enter the job description")

job_title = st.text_input("Enter job title")
st.write(f"Job Title: {job_title}")

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.size > MAX_FILE_SIZE:
            st.error(f"{file.name} exceeds 2MB limit.")
            uploaded_files = []
            break

if uploaded_files and len(uploaded_files) > 10:
    st.error("Maximum 10 resumes allowed.")

progress_bar = st.progress(0)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    for i, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        resumes.append(text)
        progress_bar.progress((i + 1) / len(uploaded_files))

    start_time = time.time()
    scores = np.round(rank_resumes(job_description, resumes) * 100, 2)
    end_time = time.time()

    results = pd.DataFrame({
        "Resume": [file.name for file in uploaded_files],
        "Score": scores,
        "Rank": [get_rank_label(score) for score in scores],
        "Email": [extract_contact_email(text) for text in resumes],
        "Resume Length": [check_resume_length(text) for text in resumes],
    })

    results = results.sort_values(by="Score", ascending=False)

    st.write(f"Ranking completed in {end_time - start_time:.2f} seconds")
    st.write(results)

    csv = results.to_csv(index=False)
    st.download_button("Download Results", csv, file_name="ranked_resumes.csv", mime="text/csv")

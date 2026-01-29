import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)

st.title("📄 AI Resume Screening & Ranking System")
st.markdown("Automatically rank resumes based on job description using NLP")


@st.cache_data
def load_data():
    return pd.read_csv("data/raw/processed_data/cleaned_resumes.csv")

df = load_data()


relevant_categories = [
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
    "DATA SCIENCE",
    "SOFTWARE",
    "DIGITAL-MEDIA"
]

filtered_df = df[
    df["Category"].str.upper().isin(relevant_categories)
].copy()


st.sidebar.header("📝 Job Description")
job_desc = st.sidebar.text_area(
    "Paste the job description here:",
    height=250,
    placeholder="Looking for a Software Engineer with Python, Java, SQL..."
)


if job_desc.strip():

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    corpus = filtered_df["cleaned_resume"].astype(str).tolist()
    corpus.append(job_desc)

    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarity_scores = cosine_similarity(
        tfidf_matrix[:-1],
        tfidf_matrix[-1]
    ).flatten()

    filtered_df["Similarity Score"] = similarity_scores

    ranked_df = filtered_df.sort_values(
        by="Similarity Score",
        ascending=False
    )


    st.subheader("🏆 Ranked Candidates")

    st.dataframe(
        ranked_df[
            ["Category", "Similarity Score", "Resume_str"]
        ].head(10),
        use_container_width=True
    )


    st.subheader("📊 Top Candidate Scores")

    chart_df = ranked_df.head(10)[["Category", "Similarity Score"]]
    chart_df = chart_df.set_index("Category")

    st.bar_chart(chart_df)

else:
    st.info("👈 Paste a job description to start ranking resumes.")

import streamlit as st
import os
import joblib
import spacy
import re

# Loading spaCy for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

PIPELINE_PATH = 'models/best_pipeline.joblib'

# Check if pipeline exists
if not os.path.exists(PIPELINE_PATH):
    st.error("Pipeline file not found! Please train your model and place it in the 'models/' folder.")
    st.stop()

# Load trained pipeline
pipeline = joblib.load(PIPELINE_PATH)

# Streamlit app
st.title("Legal Document Clause Analyzer")
st.write(
    "Upload a **.txt** legal document. "
    "I’ll extract clauses, classify their types, and highlight named entities!"
)

# File uploader
uploaded_file = st.file_uploader("Upload your legal document (.txt)", type=['txt'])

if uploaded_file is not None:
    # Read uploaded text
    file_text = uploaded_file.read().decode('utf-8')

    # Split text into clauses
    raw_clauses = re.split(r'\n+|\.\s', file_text)
    clauses = [c.strip() for c in raw_clauses if len(c.strip()) > 10]

    # original text preview
    st.subheader("Original Document Preview:")
    st.write(file_text)

    st.subheader(f"Found {len(clauses)} Clauses:")

    for clause in clauses:
        pred_type = pipeline.predict([clause])[0]
        doc = nlp(clause)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        st.markdown(f"""
        <div style="border:1px solid #ddd; padding:10px; margin-bottom:10px;">
            <strong>Clause:</strong> {clause}<br>
            <strong>Predicted Type:</strong> <span style="color:green;">{pred_type}</span><br>
            <strong>Named Entities:</strong> {ents if ents else 'None'}
        </div>
        """, unsafe_allow_html=True)

    st.success("Analysis complete!")

else:
    st.info("Please upload a **.txt** file to start the clause analysis.")

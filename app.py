import streamlit as st

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Replace this with your actual PDF splitting function
        texts = split_pdf(uploaded_file)  # your existing PDF splitting logic
        st.success("PDF processed successfully!")
    except Exception as e:
        st.error(f"PDF splitting failed: {e}")
        texts = []  # fallback so the app continues running
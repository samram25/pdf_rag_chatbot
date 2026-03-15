import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if len(documents) == 0:
            st.error("Could not read text from this PDF.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        docs = splitter.split_documents(documents)

        if len(docs) == 0:
            st.error("PDF splitting failed.")
            st.stop()

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(docs, embeddings)

        st.success("PDF processed successfully!")

        question = st.text_input("Ask a question about the PDF")

        if question:
            results = vectorstore.similarity_search(question, k=3)

            answer = ""

            for i, doc in enumerate(results):
                answer += f"Result {i+1}:\n"
                answer += doc.page_content + "\n\n"

            st.write(answer)

    except Exception as e:
        st.error(f"Error: {e}")
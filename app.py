import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set your OpenAI API key here or ensure it's set in environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

st.title("PDF RAG Chat Interface")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None and api_key:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    try:
        # Step 1: Load the PDF document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Step 2: Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Step 3: Create embeddings using OpenAI
        embeddings = OpenAIEmbeddings()

        # Step 4: Store embeddings in FAISS vector database
        vectorstore = FAISS.from_documents(docs, embeddings)

        st.success("PDF processed successfully! You can now ask questions about it.")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask a question about the PDF"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Retrieve the most relevant text chunks
            relevant_docs = vectorstore.similarity_search(prompt, k=3)

            # Combine the relevant chunks into a response
            response = "Here are the most relevant sections from the PDF:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                response += f"**Chunk {i}:**\n{doc.page_content}\n\n"

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

elif not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")

else:
    st.info("Please upload a PDF file to get started.")
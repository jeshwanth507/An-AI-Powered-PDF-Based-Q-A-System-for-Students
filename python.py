import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader

# Ensure your API key is set in environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="StudyMate: AI PDF Q&A", layout="centered")
st.title("StudyMate: AI-Powered PDF-Based Q&A System")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.from_documents(docs, embeddings)
    st.success("PDF processed! You can now ask questions.")

    user_question = st.text_input("Enter your question about the PDF content:")

    if user_question:
        with st.spinner("Generating answer..."):
            docs_relevant = db.similarity_search(user_question)
            chain = load_qa_chain(OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff")
            answer = chain.run(input_documents=docs_relevant, question=user_question)
        st.write("*Answer:*")
        st.write(answer)
else:
    st.info("Please upload a PDF file to begin.")

st.markdown("""
---
*Built for College Hackathon.*
""")
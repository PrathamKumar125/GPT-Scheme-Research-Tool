import os
import pickle
import json
import validators
from typing import List, Union

import streamlit as st
import nltk

from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader ,PyPDFLoader, DirectoryLoader


# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

# Load API key from .config
try:
    with open('.config') as f:
        config = json.load(f)
        os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
except Exception as e:
    st.error("Please create a .config file with your OpenAI API key")
    st.stop()

# Page config
st.set_page_config(page_title="Scheme Research Tool", layout="wide")

# Initialize session state
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = []
    
def validate_urls(urls):
    # Validate each URL and return only valid ones
    valid_urls = []
    for url in urls:
        if validators.url(url):
            valid_urls.append(url)
    return valid_urls

def get_scheme_summary(text):
    # Construct prompt for scheme summary generation using OpenAI
    prompt = """Summarize the scheme information in the following categories:
    1. Benefits
    2. Application Process
    3. Eligibility
    4. Required Documents"""
    
    llm = OpenAI()
    summary = llm.predict(prompt + "\n\nText: " + text)
    return summary

def process_urls(urls: List[str]) -> Union[tuple, None]:
    try:
        valid_urls = validate_urls(urls)
        if not valid_urls:
            raise ValueError("No valid URLs provided")
        
        documents = []
        for url in valid_urls:
            if url.lower().endswith('.pdf'):
                import requests
                import tempfile
                
                response = requests.get(url, verify=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())
                os.unlink(temp_path)
            else:
                loader = UnstructuredURLLoader(
                    urls=[url],
                    continue_on_failure=True,
                    headers={"User-Agent": "Mozilla/5.0"},
                    ssl_verify=False
                )
                documents.extend(loader.load())
        
        if not documents:
            raise ValueError("No content could be extracted from URLs")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        with open("faiss_store_openai.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        
        return (vectorstore, documents)
        
    except Exception as e:
        st.error(f"Error processing URLs: {str(e)}")
        return (None, None)

def main():
    st.title("Scheme Research Tool")
    
    with st.sidebar:
        # Sidebar UI for URL input and processing
        st.header("Input URLs")
        url_input = st.text_area("Enter URLs (one per line)")
        file_upload = st.file_uploader("Or upload a file with URLs", type=['txt'])
        
        if file_upload:
            url_input = file_upload.getvalue().decode()
        
        process_button = st.button("Process URLs")

    if process_button and url_input:
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        with st.spinner("Processing URLs..."):
            vectorstore, data = process_urls(urls)
            if vectorstore and data:
                st.session_state.processed_urls.extend(urls)
                summary = get_scheme_summary(str(data))
                st.session_state.summary = summary
                st.success("URLs processed successfully!")
            else:
                st.error("Failed to process URLs")

    # Main content area for displaying results
    if hasattr(st.session_state, 'summary'):
        st.header("Scheme Summary")
        st.write(st.session_state.summary)
        
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the schemes:")

    if question and st.session_state.processed_urls:
        try:
            with open("faiss_store_openai.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=OpenAI(),
                retriever=vectorstore.as_retriever()
            )
            
            result = qa_chain({"question": question, "chat_history": []})
            
            st.subheader("Answer:")
            st.write(result["answer"])
            
            st.subheader("Sources:")
            for url in st.session_state.processed_urls:
                st.write(url)
                
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
    
    elif not st.session_state.processed_urls:
        st.info("Please process URLs first before asking questions.")

if __name__ == "__main__":
    main()
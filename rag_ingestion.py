# -*- coding: utf-8 -*-
import os
import sys
import pickle
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Manually add the site-packages directory to the Python path.
# This is a robust way to ensure modules are found within the conda environment.
try:
    conda_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    site_packages_path = os.path.join(conda_env_path, 'lib', 'python' + '.'.join(str(v) for v in sys.version_info[:2]), 'site-packages')
    if os.path.exists(site_packages_path) and site_packages_path not in sys.path:
        sys.path.append(site_packages_path)
    
    # Check if the modules are now importable after the path modification
    import langchain
    import langchain_community

except Exception as e:
    print(f"Failed to set Python path for LangChain: {e}")
    print("This might be a temporary issue. Continuing with execution.")

def ingest_data(file_path="knowledge_base.txt", output_file="faiss_index.bin"):
    """
    Loads a text file, splits the document, creates vector embeddings, and saves the FAISS vector store.

    Args:
        file_path (str): Path to the knowledge base file.
        output_file (str): The filename for the saved FAISS vector store.
    """
    if not os.path.exists(file_path):
        print(f"Error: Knowledge base file '{file_path}' not found. Please create the file with content first.")
        return

    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path)
    documents = loader.load()
    print("Document loaded successfully.")

    print("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f"Split document into {len(docs)} chunks.")

    print("Creating vector embeddings. This may take a moment...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    print("Vector store created successfully.")

    print(f"Saving vector store to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(vector_store, f)
    print("Vector store saved successfully.")

if __name__ == "__main__":
    ingest_data()
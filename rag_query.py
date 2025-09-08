import os
import pickle
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

model_path = os.path.join(os.getcwd(), "llama-2-7b-chat.Q2_K.gguf")

def load_vector_store(file_name="faiss_index.bin"):
    """
    Loads a FAISS vector store from a local file.

    Args:
        file_name (str): The name of the vector store file.

    Returns:
        FAISS: The loaded FAISS vector store object.
    """
    if not os.path.exists(file_name):
        print(f"Error: The file {file_name} was not found. Please run rag_ingestion.py first.")
        return None
    
    print(f"Loading vector store from {file_name}...")
    with open(file_name, "rb") as f:
        vector_store = pickle.load(f)
    print("Vector store loaded successfully.")
    return vector_store

def load_local_llm(model_path):
    """
    Loads a local LlamaCpp LLM model.

    Args:
        model_path (str): The path to the local LLM model file.

    Returns:
        LlamaCpp: The LlamaCpp LLM object.
    """
    print(f"Loading local LLM from {model_path}...")
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max_tokens=2000,
        n_ctx=4096,
        top_p=1,
        n_gpu_layers=0,  # Set this value to > 0 to use GPU acceleration if you have a GPU
        verbose=False,
    )
    print("LLM loaded successfully.")
    return llm

def run_query_loop(llm, vector_store):
    """
    Runs a loop that accepts user input and generates RAG answers.

    Args:
        llm (LlamaCpp): The loaded LLM model.
        vector_store (FAISS): The loaded vector store.
    """
    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    
    print("\n---------------------------------------------------------")
    print("RAG System is ready! You can now ask questions about the knowledge base.")
    print("Type 'exit' to quit.")
    print("---------------------------------------------------------")

    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            print("Exiting RAG System. Goodbye!")
            break
        
        print("Searching and generating response...")
        response = qa_chain.invoke({"query": query})
        print("\n------------------ Response ------------------")
        print(response['result'])
        print("----------------------------------------------")

if __name__ == '__main__':
    model_path = "llama-2-7b-chat.Q2_K.gguf"
    
    vector_store = load_vector_store()
    
    llm = load_local_llm(model_path)
    
    if vector_store and llm:
        run_query_loop(llm, vector_store)
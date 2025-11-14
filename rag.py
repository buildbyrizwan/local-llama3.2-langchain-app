import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import Docx2txtLoader

# Function to load all documents

def load_all_documents(folder_path: str):
    """Load all .txt, .pdf, and .docx files from a folder."""
    all_docs = []
    folder = Path(folder_path)

    for file in folder.glob("**/*"):
        if file.suffix.lower() == ".txt":
            loader = TextLoader(str(file))
        elif file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(file))
        else:
            continue

        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


st.title("🦙 LangChain RAG App with LLAMA2")
st.write("Load unstructured files → Embed → Store in Chroma → Ask Questions to Llama2")

data_path = r"C:\Users\ext-Dakshak\Documents\chatproj\pdf"

query = st.text_input("🔍 Ask your question:")

if st.button("Run RAG"):
    if not query.strip():
        st.warning(" Please enter a query.")
    else:
        with st.spinner("Processing your files..."):
            # 1 Load all documents
            documents = load_all_documents(data_path)
            if not documents:
                st.error("No supported files found in the folder.")
                st.stop()

            st.success(f"Loaded {len(documents)} documents.")

            # 2 Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            st.write(f"Created {len(texts)} chunks.")

            # 3 Generate embeddings
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            # 4 Store in ChromaDB
            vector_db = Chroma.from_documents(texts, embedding_function, persist_directory="./chroma_store")
            vector_db.persist()
            st.success(" All documents embedded and stored in ChromaDB.")

            # 5 Retrieve top chunks
            results = vector_db.similarity_search(query, k=3)
            context = "\n".join([r.page_content for r in results])

            st.write("### Retrieved Context")
            st.info(context)

            # 6 Send context + query to LLAMA2 (local)
            llm = OllamaLLM(model="llama3.2")
            prompt = f"""
            You are a helpful assistant. Use the following context to answer the question.
            Context:
            {context}
            Question:
            {query}
            Answer:
            """

            response = llm.invoke(prompt)
            st.subheader(" LLM Response:")
            st.success(response)

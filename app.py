import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG PDF Chat", layout="wide")

st.title("📚 Chat with your PDF (RAG System)")

# ----------- SESSION STATE -----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ----------- FILE UPLOAD -----------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_path = f"temp_{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # ----------- LOAD & PROCESS -----------
    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        embeddings = MistralAIEmbeddings(model="mistral-embed")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever

    st.success("RAG system ready!")

# ----------- QUERY INPUT -----------
query = st.text_input("Ask a question from your PDF:")

if query and st.session_state.retriever:
    with st.spinner("Thinking..."):

        docs = st.session_state.retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
                ),
                (
                    "human",
                    """Context:
{context}

Question:
{question}
"""
                )
            ]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        llm = ChatMistralAI(model="mistral-small-2506")
        response = llm.invoke(final_prompt)

        st.subheader("Answer:")
        st.write(response.content)

        with st.expander("🔍 Retrieved Context"):
            st.write(context)

elif query:
    st.warning("Upload a PDF first.")
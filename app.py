import os
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_config import get_embeddings, get_llm

load_dotenv()

st.set_page_config(page_title="LearnMate AI", page_icon="📚", layout="wide")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)

# ----------- SESSION STATE -----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("doc_info", None)
st.session_state.setdefault("processed_file", None)


# ----------- HELPERS -----------
def index_pdf(uploaded_file):
    """Read, split and embed the PDF. Returns (retriever, info) or (None, None)."""
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    started = time.perf_counter()
    with st.status("Indexing your PDF…", expanded=True) as status:
        try:
            st.write("📖 Reading pages…")
            docs = PyPDFLoader(file_path).load()

            st.write("✂️ Splitting into chunks…")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            if not chunks:
                status.update(label="No readable text found", state="error")
                st.error("This PDF has no extractable text. Scanned PDFs (images of pages) are not supported.")
                return None, None

            st.write("🧠 Creating embeddings…")
            embeddings = get_embeddings()

            # Start from an empty collection so answers never mix two documents
            store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
            store.delete_collection()
            store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

            batch_size = 256
            bar = st.progress(0.0, text=f"Embedded 0/{len(chunks)} chunks")
            for i in range(0, len(chunks), batch_size):
                store.add_documents(chunks[i : i + batch_size])
                done = min(i + batch_size, len(chunks))
                bar.progress(done / len(chunks), text=f"Embedded {done}/{len(chunks)} chunks")

            retriever = store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
            )
            info = {
                "name": uploaded_file.name,
                "pages": len(docs),
                "chunks": len(chunks),
                "seconds": time.perf_counter() - started,
            }
            status.update(label="PDF ready", state="complete", expanded=False)
            return retriever, info
        except Exception as e:
            status.update(label="Indexing failed", state="error")
            st.error(f"Could not index this PDF: {e}")
            return None, None
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


def show_sources(sources):
    with st.expander(f"🔍 Sources ({len(sources)})"):
        for n, s in enumerate(sources, 1):
            st.markdown(f"**Source {n} · page {s['page']}**")
            st.caption(s["text"])


def friendly_error(e):
    text = str(e)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "The AI service is rate-limited right now. Please wait a minute and try again."
    return f"Sorry, something went wrong: {text}"


# ----------- SIDEBAR: UPLOAD -----------
with st.sidebar:
    st.markdown("## 📚 LearnMate AI")
    st.caption("Ask questions about your PDFs. Answers come only from your document.")
    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Process each PDF only once: Streamlit re-runs this script on every interaction.
if uploaded_file and st.session_state.processed_file != uploaded_file.name:
    retriever, info = index_pdf(uploaded_file)
    if retriever:
        st.session_state.retriever = retriever
        st.session_state.doc_info = info
        st.session_state.processed_file = uploaded_file.name
        st.session_state.messages = []
        st.toast("Your PDF is ready — ask away!", icon="✅")

# ----------- SIDEBAR: DOCUMENT INFO -----------
info = st.session_state.doc_info
with st.sidebar:
    if info:
        st.divider()
        st.markdown(f"**📄 {info['name']}**")
        c1, c2 = st.columns(2)
        c1.metric("Pages", info["pages"])
        c2.metric("Chunks", info["chunks"])
        st.caption(f"Indexed in {info['seconds']:.0f}s")
        if st.button("🗑️ Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()

# ----------- MAIN: EMPTY STATE OR CHAT -----------
if not st.session_state.retriever:
    st.title("Chat with your PDF")
    st.write("Upload a document in the sidebar to get started.")
    steps = [
        ("1️⃣", "Upload", "Drop in a PDF — textbooks, papers, notes."),
        ("2️⃣", "Ask", "Type a question in plain language."),
        ("3️⃣", "Get answers", "Replies are grounded in your document, with the sources shown."),
    ]
    for col, (icon, title, text) in zip(st.columns(3), steps):
        with col, st.container(border=True, height=170):
            st.markdown(f"### {icon} {title}")
            st.caption(text)
else:
    st.title(f"📄 {info['name']}")
    if not st.session_state.messages:
        st.info("Your document is indexed. Ask your first question below.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

question = st.chat_input(
    "Ask something about your document…" if st.session_state.retriever else "Upload a PDF to start",
    disabled=not st.session_state.retriever,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                docs = st.session_state.retriever.invoke(question)
                context = "\n\n".join(d.page_content for d in docs)
                reply = get_llm().invoke(PROMPT.invoke({"context": context, "question": question}))
            sources = [
                {"page": d.metadata.get("page", 0) + 1, "text": d.page_content}
                for d in docs
            ]
            st.markdown(reply.text)
            show_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply.text, "sources": sources}
            )
        except Exception as e:
            st.error(friendly_error(e))
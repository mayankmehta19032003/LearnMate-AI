from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Deep learning is a subset of machine learning that uses neural networks with many layers.",
        metadata={"source": "dl_book", "page": 1}
    ),
    Document(
        page_content="Docker is a containerization platform used to package applications with their dependencies.",
        metadata={"source": "devops_notes", "page": 5}
    ),
    Document(
        page_content="Vector databases store embeddings and allow similarity search using distance metrics like cosine similarity.",
        metadata={"source": "rag_guide", "page": 2}
    )
]


embeddings_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents= docs,
    embedding=embeddings_model,
    persist_directory="chroma-db", 
)

result = vector_store.similarity_search("what is vectore store?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
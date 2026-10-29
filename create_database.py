#load pdf
#split into chunks
#create the emmbedings
#store in chroma
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_config import get_embeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

#-------loader------------
data = PyPDFLoader("document loader/deeplearning.pdf")
docs = data.load()

#--------text splitter / chunking---------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

#--------emmbedding-----------------------
embeddings_model = get_embeddings()

#----------load data in chroma----------
vector_store = Chroma.from_documents(
    documents= chunks,
    embedding=embeddings_model,
    persist_directory="chroma_db", 
)

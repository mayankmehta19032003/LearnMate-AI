import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
    # Runs on your own CPU inside the container: no API calls, no quota
    return FastEmbedEmbeddings(
        model_name=EMBED_MODEL,
        cache_dir=os.getenv("FASTEMBED_CACHE_PATH"),
    )


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", timeout=60, max_retries=2)
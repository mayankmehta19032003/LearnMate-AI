from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", timeout=60, max_retries=2)
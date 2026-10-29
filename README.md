# LearnMate AI 📚

Chat with your PDFs. Upload a PDF, ask questions, and get answers based only on the document (RAG: Retrieval-Augmented Generation).

**Tech:** Python · Streamlit · LangChain · ChromaDB · Google Gemini · Docker

## Quick start (Docker)

**You need:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/mayankmehta19032003/LearnMate-AI.git
cd LearnMate-AI
```

Create a file named `.env` containing:

```
GOOGLE_API_KEY=your_key_here
```

Start the app:

```bash
docker compose up --build
```

Open http://localhost:8501, upload a PDF and ask a question.
Stop with `Ctrl+C`, then `docker compose down`.

## How it works

1. The PDF is split into chunks.
2. Each chunk is turned into an embedding (a list of numbers) and stored in ChromaDB.
3. Your question is embedded too, and the most similar chunks are retrieved.
4. Gemini answers using only those chunks.

# LearnMate AI 📚

![Publish Docker image](https://github.com/mayankmehta19032003/LearnMate-AI/actions/workflows/docker-publish.yml/badge.svg)

Chat with your PDFs. Upload a PDF, ask questions, and get answers based only on the document (RAG: Retrieval-Augmented Generation).

**Tech:** Python · Streamlit · LangChain · ChromaDB · Google Gemini · Docker · GitHub Actions

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A free [Gemini API key](https://aistudio.google.com/apikey)

## Option 1: Run the published image (no clone needed)

The image is built and published automatically by GitHub Actions on every change to `main`.

```bash
docker run --rm -p 8501:8501 -e GOOGLE_API_KEY=your_key_here ghcr.io/mayankmehta19032003/learnmate-ai:latest
```

Open http://localhost:8501, upload a PDF and ask a question. Stop with `Ctrl+C`.

The first run downloads the image (about 500 MB). Later runs start in seconds. To download it ahead of time:

```bash
docker pull ghcr.io/mayankmehta19032003/learnmate-ai:latest
```

## Option 2: Build it yourself from source

```bash
git clone https://github.com/mayankmehta19032003/LearnMate-AI.git
cd LearnMate-AI
```

Create a file named `.env` containing:

```
GOOGLE_API_KEY=your_key_here
```

Then start the app:

```bash
docker compose up --build
```

Open http://localhost:8501. Stop with `Ctrl+C`, then `docker compose down`.

## How it works

1. The PDF is split into chunks.
2. Each chunk is turned into an embedding (a list of numbers) and stored in ChromaDB.
3. Your question is embedded too, and the most similar chunks are retrieved.
4. Gemini answers using only those chunks.

## Notes

- Your API key is never stored in the image. It is passed in when the container starts.
- The Gemini free tier has rate limits, so use smaller PDFs for the quickest experience.

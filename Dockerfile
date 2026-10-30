FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Must come AFTER pip install: downloads the embedding model into the image
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=os.environ['FASTEMBED_CACHE_PATH'])"

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.maxUploadSize=500"]
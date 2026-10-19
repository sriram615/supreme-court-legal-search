FROM python:3.11-slim

# Hugging Face Docker Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
USER user

# CPU-only torch (the default PyPI wheel on Linux pulls ~2GB of CUDA libraries).
# gradio / spacy / pdfplumber / pypdf are left out: app.py skips the optional
# /gradio mount when gradio is missing, and the others are ingestion-only.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && grep -vE '^(torch==|spacy$|pdfplumber$|pypdf$|gradio$|#|$)' requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt

# Bake the embedding model into the image so cold starts don't download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user app.py search_engine.py utils.py validator.py agentic_layer.py ./
COPY --chown=user frontend ./frontend
COPY --chown=user LAWdata_Corpus_2024 ./LAWdata_Corpus_2024

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

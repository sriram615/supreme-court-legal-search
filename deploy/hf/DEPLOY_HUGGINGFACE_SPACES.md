# Free 100% Host Deployment: Hugging Face Spaces (Docker)

Hugging Face Spaces offers **free Docker application hosting** with **2 vCPUs, 16 GB RAM**, no credit card required, and no sleep timeout restrictions for web apps.

---

## Quick Start (Automated Upload)

### Step 1: Run the Deployment Script

From the repository root:
```bash
.venv311/bin/python deploy/hf/deploy_hf.py
```

### Step 2: Provide Your Free Hugging Face Access Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **Create new token** (Role: **Write**)
3. Paste the token when prompted by the script.

### Step 3: View Your Live App!

The script will automatically:
- Create your Hugging Face Space (`https://huggingface.co/spaces/<your-username>/legal-search-engine`)
- Upload a Docker-flavoured `README.md`, `deploy/hf/Dockerfile`, the server code, the frontend, and only the
  runtime corpus files (`legal_precedents.index`, `metadata_map.json`, `citation_map.json`, `manifest.json`;
  roughly 80MB). Everything else (`.sesskey`, `data/`, logs, reports, the Legal-BERT model, tests) is never uploaded
  because the script uses an allowlist.
- Set the `GROQ_API_KEY` secret (if you enter one) and an `LLM_MODEL` variable
- Trigger the Docker build on Hugging Face servers

The first build takes several minutes (CPU torch and the embedding model are installed into the image).
The app listens on port 7860. When it is live it is publicly reachable at:
`https://<your-username>-legal-search-engine.hf.space`

---

## Hardware Specifications (Free Tier)

| Spec | Value |
|---|---|
| **CPU** | 2 vCPUs (Intel Xeon / AMD EPYC) |
| **RAM** | 16 GB |
| **Disk** | 50 GB |
| **Cost** | 100% Free |
| **Timeout** | Always-on / auto-resume on request |


---

## Notes and limits

- The Space is **public** by default, and the API has no authentication or rate limit. Anyone can use your Groq
  quota. Set the Space to private, or add rate limiting, before sharing the link.
- If `GROQ_API_KEY` is not set, the referee reports "Verification unavailable" and `/api/v1/agentic-analyze` returns
  retrieval results with no synthesized answer.
- Ollama is not available on Spaces, so the referee always uses the Groq fallback there.
- Submitted briefs are sent to Groq's API.
- Not verified end to end: the image build itself has not been run (no Docker daemon was available). The app code
  was run against exactly the allowlisted files and served `/`, `/api/v1/health` and `/api/v1/analyze`.

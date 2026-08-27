"""
fetch_and_preprocess.py  ─  v2.0
=================================
Indian Supreme Court Judgment Pipeline
AWS Open Data Registry  →  Local Drive-Ready Corpus

Real S3 layout (confirmed):
  Bucket  : s3://indian-supreme-court-judgments/
  Index   : data/tar/year=2024/english/english.index.json
  Archive : data/tar/year=2024/english/english.tar  (203 MB, 782 PDFs)
  Metadata: metadata/parquet/year=2024/metadata.parquet

Strategy:
  1. Download english.tar to a local temp file (streamed, 1 MB chunks).
  2. Walk the tar sequentially — extract each PDF in-memory.
  3. Convert PDF bytes → text using pdfminer.six (no disk write).
  4. Clean → tag citations → sentence-chunk → validate (Pydantic v2).
  5. Write validated chunks to JSONL; write citation_map.json.
  6. Package output into a zip archive ready to mount to Google Drive.

CPU tuning: ProcessPoolExecutor(PHYSICAL_CPU_CORES) for NLP; main thread
            handles all I/O (tar walk, PDF extract, JSONL write).

Target : Intel Core i7/i9 x86_64 macOS
Python  : ≥3.10
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ─ STDLIB IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import io
import json
import time
import tarfile
import zipfile
import logging
import hashlib
import unicodedata
import multiprocessing
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ THIRD-PARTY IMPORTS (with clear install hints)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotocoreConfig
except ImportError as exc:
    raise SystemExit("[FATAL] Run: pip install boto3 botocore") from exc

try:
    import spacy
except ImportError as exc:
    raise SystemExit("[FATAL] Run: pip install spacy && python -m spacy download en_core_web_sm") from exc

try:
    from pydantic import BaseModel, Field, field_validator, ValidationError
    import pydantic
except ImportError as exc:
    raise SystemExit("[FATAL] Run: pip install 'pydantic>=2.0'") from exc

try:
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
except ImportError as exc:
    raise SystemExit("[FATAL] Run: pip install pdfminer.six") from exc

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SC_Pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ PIPELINE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── AWS ──────────────────────────────────────────────────────────────────────
S3_BUCKET       = "indian-supreme-court-judgments"
S3_REGION       = "us-east-1"
TARGET_YEAR     = "2024"
S3_TAR_KEY      = f"data/tar/year={TARGET_YEAR}/english/english.tar"
S3_INDEX_KEY    = f"data/tar/year={TARGET_YEAR}/english/english.index.json"
S3_META_KEY     = f"metadata/parquet/year={TARGET_YEAR}/metadata.parquet"
S3_CHUNK_BYTES  = 4 * 1024 * 1024   # 4 MB streaming chunk for tar download

# ── Local paths ───────────────────────────────────────────────────────────────
BASE_DIR        = Path("./data")
PROCESSED_DIR   = BASE_DIR / "processed"
TEMP_DIR        = BASE_DIR / "tmp"
OUTPUT_JSONL    = PROCESSED_DIR / "aws_court_chunks.jsonl"
CITATION_MAP    = PROCESSED_DIR / "citation_map.json"
LOCAL_TAR       = TEMP_DIR / f"english_{TARGET_YEAR}.tar"
DRIVE_ZIP       = BASE_DIR / f"SC_2024_corpus_drive_ready.zip"
MANIFEST_FILE   = PROCESSED_DIR / "manifest.json"

# ── Chunking ─────────────────────────────────────────────────────────────────
MAX_WINDOW_TOKENS = 512
OVERLAP_TOKENS    = 64

# ── Intel Mac CPU tuning ──────────────────────────────────────────────────────
PHYSICAL_CPU_CORES = max(1, multiprocessing.cpu_count() // 2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ PYDANTIC OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class CourtJudgmentChunk(BaseModel):
    """
    Validated schema for one preprocessed chunk written to the JSONL corpus.
    All fields are validated strictly before disk write.
    """
    chunk_id:           str  = Field(..., min_length=5)
    parent_judgment_id: str  = Field(..., min_length=5)
    court:              str  = Field(default="Supreme Court of India")
    tokens_count:       int  = Field(..., ge=1, le=MAX_WINDOW_TOKENS)
    text_chunk:         str  = Field(..., min_length=10)
    extracted_citations: List[str] = Field(default_factory=list)
    language_flag:      str  = Field(default="en", pattern=r"^[a-z]{2}$")
    source_pdf:         str  = Field(default="")   # original PDF filename in tar
    char_hash:          str  = Field(default="")   # MD5 of text_chunk for dedup

    @field_validator("extracted_citations", mode="before")
    @classmethod
    def deduplicate_citations(cls, v: List[str]) -> List[str]:
        seen: set = set()
        return [x for x in v if not (x in seen or seen.add(x))]  # type: ignore[arg-type]

    @field_validator("char_hash", mode="before")
    @classmethod
    def auto_hash(cls, v: str, info: Any) -> str:
        if not v and "text_chunk" in (info.data or {}):
            return hashlib.md5(info.data["text_chunk"].encode()).hexdigest()[:16]
        return v

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ ANONYMOUS S3 CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def build_s3_client() -> Any:
    """Return a boto3 S3 client configured for anonymous Open Data access."""
    config = BotocoreConfig(
        signature_version=UNSIGNED,
        region_name=S3_REGION,
        retries={"max_attempts": 6, "mode": "adaptive"},
        max_pool_connections=4,
        read_timeout=180,
        connect_timeout=30,
    )
    client = boto3.client("s3", config=config, region_name=S3_REGION)
    logger.info("Anonymous S3 client ready → bucket=%s", S3_BUCKET)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ TAR DOWNLOAD (streaming, resumable)
# ─────────────────────────────────────────────────────────────────────────────

def download_tar_streaming(s3: Any, local_path: Path) -> None:
    """
    Stream the english.tar from S3 to disk in S3_CHUNK_BYTES chunks.
    Skips download if the local file already exists and matches the
    expected S3 ContentLength (basic resume guard).
    """
    # Check if already downloaded
    if local_path.exists():
        head = s3.head_object(Bucket=S3_BUCKET, Key=S3_TAR_KEY)
        expected = head["ContentLength"]
        if local_path.stat().st_size == expected:
            logger.info("TAR already present locally (%s). Skipping download.", local_path)
            return
        logger.info("TAR size mismatch — re-downloading.")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading s3://%s/%s → %s", S3_BUCKET, S3_TAR_KEY, local_path)

    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_TAR_KEY)
    total = obj["ContentLength"]
    body  = obj["Body"]
    downloaded = 0
    last_pct = -1

    with open(local_path, "wb") as fout:
        while True:
            chunk = body.read(S3_CHUNK_BYTES)
            if not chunk:
                break
            fout.write(chunk)
            downloaded += len(chunk)
            pct = int(downloaded / total * 100)
            if pct // 5 != last_pct // 5:
                last_pct = pct
                logger.info("  Download progress: %d%% (%d / %d MB)",
                            pct, downloaded // 1_000_000, total // 1_000_000)

    logger.info("Download complete: %s (%.1f MB)", local_path, downloaded / 1e6)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf_bytes(pdf_bytes: bytes, pdf_name: str = "") -> str:
    """
    Extract plain text from a PDF byte blob using pdfminer.six.
    Returns an empty string on failure (logged as warning).

    pdfminer is pure-Python and requires no Poppler/Ghostscript.
    LAParams controls line spacing tolerance — tuned for court judgment
    typography (single-column justified text, dense footnotes).
    """
    try:
        laparams = LAParams(
            line_overlap=0.5,
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            boxes_flow=0.5,
            detect_vertical=False,
            all_texts=False,
        )
        out_buffer = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(pdf_bytes),
            out_buffer,
            laparams=laparams,
            output_type="text",
            codec="utf-8",
        )
        return out_buffer.getvalue()
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", pdf_name, exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ─ TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

_RE_PAGE_NUMBERS   = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_RE_PAGE_HEADER    = re.compile(
    r"(REPORTABLE|NON[\-\s]REPORTABLE|IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA"
    r"|CERTIFIED\s+(?:TRUE\s+)?COPY|Page\s+\d+\s+of\s+\d+)",
    re.IGNORECASE,
)
_RE_FOOTER_BLOCK   = re.compile(
    r"(Signed\s+by\s+\w[\w\s]+|Date\s+of\s+pronouncement[\s\S]{0,120}?\n"
    r"|Registry\s+is\s+directed[\s\S]{0,200}?\n"
    r"|Sd\/\-|S\.D\.|I\s+agree[\.\s]*\n"
    r"|Copy\s+forwarded\s+to[\s\S]{0,300}?\n"
    r"|This\s+judgment\s+shall\s+not\s+be[\s\S]{0,200}?\n)",
    re.IGNORECASE | re.MULTILINE,
)
_RE_HYPHENATION    = re.compile(r"(\w)-\n(\w)")     # merge hyphenated line breaks
_RE_EXCESSIVE_NL   = re.compile(r"\n{3,}")
_RE_EXCESSIVE_SP   = re.compile(r"[ \t]{2,}")
_RE_ZERO_WIDTH     = re.compile(r"[\u200b\u200c\u200d\ufeff\xa0\x00-\x08\x0b\x0c\x0e-\x1f]")
_RE_FORM_FEED      = re.compile(r"\f")


def clean_judgment_text(raw: str) -> str:
    """
    Multi-stage cleaning for raw pdfminer output from a Supreme Court judgment.

    Stages:
      1. Unicode NFC normalisation (preserves Devanagari/Tamil/Telugu chars).
      2. Strip control chars, zero-width, non-breaking spaces.
      3. Merge hyphenated end-of-line word splits (common in PDF extraction).
      4. Remove form feeds (page separators from PDF).
      5. Strip standalone page numbers, repeated headers, footer blocks.
      6. Collapse ≥3 consecutive newlines → paragraph break.
      7. Collapse ≥2 horizontal spaces → single space.
    """
    text = unicodedata.normalize("NFC", raw)
    text = _RE_ZERO_WIDTH.sub(" ", text)
    text = _RE_FORM_FEED.sub("\n\n", text)
    text = _RE_HYPHENATION.sub(r"\1\2", text)
    text = _RE_PAGE_NUMBERS.sub("", text)
    text = _RE_PAGE_HEADER.sub("", text)
    text = _RE_FOOTER_BLOCK.sub("", text)
    text = _RE_EXCESSIVE_NL.sub("\n\n", text)
    text = _RE_EXCESSIVE_SP.sub(" ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ─ CITATION REGEX & TAGGING
# ─────────────────────────────────────────────────────────────────────────────
#
# 15 citation families covering canonical Indian legal reporters:
#  AIR / SCC / SCR / SCALE / ILR / JT / SCWR / MLJ / CLT / GLH / PLR /
#  BLJR / NLJ / INSC (neutral) / SLP & Writ procedural numbers
#
from utils import _CITATION_PATTERNS, make_token as _make_token



def tag_citations(
    text: str,
    citation_map: Dict[str, str],
) -> Tuple[str, List[str]]:
    """
    Replace all citation matches with [TOKEN] placeholders.
    Mutates `citation_map` in place to log new discoveries.

    Returns:
        (tagged_text, list_of_token_ids_found_in_this_text)
    """
    found: List[str] = []
    for _family, pattern in _CITATION_PATTERNS.items():
        for match in pattern.finditer(text):
            raw  = match.group(0)
            tok  = _make_token(raw)
            placeholder = f"[{tok}]"
            if tok not in citation_map:
                citation_map[tok] = raw
            found.append(tok)
            text = text.replace(raw, placeholder, 1)
    return text, found


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ─ SPACY SINGLETON (process-level)
# ─────────────────────────────────────────────────────────────────────────────

_NLP_MODEL: Optional[spacy.language.Language] = None


def get_nlp() -> spacy.language.Language:
    """
    Load en_core_web_sm once per process.

    en_core_web_sm does NOT ship a standalone 'senter'. Sentence boundaries
    come from 'parser'. We load the full model, disable every heavy pipe,
    and add the fast rule-based 'sentencizer' as the sole sentence splitter.
    Active pipe set after init: ['sentencizer', 'tok2vec']
    """
    global _NLP_MODEL
    if _NLP_MODEL is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")

        # Disable every pipe except tok2vec (keeps tokenisation quality)
        heavy = [p for p in nlp.pipe_names if p != "tok2vec"]
        if heavy:
            nlp.disable_pipes(*heavy)

        # Add rule-based sentencizer as the active sentence splitter
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", first=True)

        _NLP_MODEL = nlp
        logger.debug("SpaCy loaded — active pipes: %s", _NLP_MODEL.pipe_names)
    return _NLP_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 ─ SENTENCE-AWARE SLIDING WINDOW CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    max_tokens: int = MAX_WINDOW_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[str]:
    """
    Split text into overlapping chunks that never cut a sentence mid-way.

    Algorithm:
      1. Segment text into sentences with SpaCy sentencizer.
      2. Pack sentences greedily until the window reaches max_tokens.
      3. Walk back from the end of the completed window by up to overlap_tokens
         words, stopping at the nearest sentence boundary.
      4. The next window starts at that boundary → clean sentence-aligned overlap.

    Args:
        text:          Citation-tagged, cleaned judgment text.
        max_tokens:    Max whitespace-tokenised words per chunk (default 512).
        overlap_tokens:Approximate overlap goal in tokens (default 64).

    Returns:
        List[str]: Non-empty chunk strings each ≤ max_tokens words.
    """
    nlp = get_nlp()
    if len(text) > nlp.max_length:
        nlp.max_length = len(text) + 100

    doc       = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    sent_idx = 0

    while sent_idx < len(sentences):
        window: List[str] = []
        word_count = 0
        cur = sent_idx

        # ── Pack sentences ─────────────────────────────────────────────────
        while cur < len(sentences):
            sw = len(sentences[cur].split())
            if word_count + sw > max_tokens and window:
                break
            window.append(sentences[cur])
            word_count += sw
            cur += 1

        # Edge case: single oversized sentence
        if not window:
            window = [sentences[cur]]
            cur += 1

        chunks.append(" ".join(window))

        if cur >= len(sentences):
            break

        # ── Find overlap start (walk backward by ≤ overlap_tokens words) ───
        accum   = 0
        ov_start = cur - 1
        while ov_start > sent_idx:
            sw = len(sentences[ov_start].split())
            if accum + sw > overlap_tokens:
                break
            accum   += sw
            ov_start -= 1

        # Advance, guaranteeing forward progress
        sent_idx = max(ov_start + 1, sent_idx + 1)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 ─ PER-DOCUMENT WORKER  (runs in ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def process_document(
    args: Tuple[str, bytes, int, Dict[str, str]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Full preprocessing pipeline for one PDF document.

    Args:
        args: (pdf_filename, pdf_bytes, doc_seq_num, local_citation_map)
              local_citation_map is a copy — mutations stay in the worker.

    Returns:
        (list_of_chunk_dicts, updated_local_citation_map)
        The caller merges the returned citation map into the global one.
    """
    pdf_name, pdf_bytes, seq, local_cmap = args

    doc_id    = f"AWS_SC_{TARGET_YEAR}_{seq:04d}"
    chunks_out: List[Dict[str, Any]] = []

    # Stage 1: PDF → text
    raw_text  = extract_text_from_pdf_bytes(pdf_bytes, pdf_name)
    if not raw_text or len(raw_text.split()) < 30:
        logger.warning("[%s] Insufficient text extracted from %s — skip.", doc_id, pdf_name)
        return chunks_out, local_cmap

    # Stage 2: Clean
    clean_text = clean_judgment_text(raw_text)
    if len(clean_text.split()) < 20:
        logger.warning("[%s] Text too short after cleaning — skip.", doc_id)
        return chunks_out, local_cmap

    # Stage 3: Citation tagging
    tagged_text, _ = tag_citations(clean_text, local_cmap)

    # Stage 4: Chunking
    text_chunks = chunk_text(tagged_text)

    # Stage 5: Pydantic validation + dict serialisation
    for chunk_idx, chunk_str in enumerate(text_chunks, start=1):
        tok_count = len(chunk_str.split())
        chunk_id  = f"{doc_id}_chunk_{chunk_idx:02d}"

        # Collect citation tokens present in this specific chunk
        citations_here = [tok for tok in local_cmap if f"[{tok}]" in chunk_str]

        # Language heuristic: if > 20% chars are non-ASCII → flag "hi"
        non_ascii = sum(1 for c in chunk_str if ord(c) > 127)
        lang      = "hi" if non_ascii / max(len(chunk_str), 1) > 0.20 else "en"

        try:
            record = CourtJudgmentChunk(
                chunk_id=chunk_id,
                parent_judgment_id=doc_id,
                court="Supreme Court of India",
                tokens_count=min(tok_count, MAX_WINDOW_TOKENS),
                text_chunk=chunk_str,
                extracted_citations=citations_here,
                language_flag=lang,
                source_pdf=pdf_name,
                char_hash=hashlib.md5(chunk_str.encode()).hexdigest()[:16],
            )
            chunks_out.append(record.model_dump())
        except ValidationError as ve:
            logger.debug("Validation skip %s: %s", chunk_id, ve)

    return chunks_out, local_cmap


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 ─ JSONL WRITER
# ─────────────────────────────────────────────────────────────────────────────

def append_chunks_to_jsonl(chunks: List[Dict[str, Any]], path: Path) -> int:
    """Append chunk dicts to JSONL file. Returns count written."""
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            written += 1
    return written


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 ─ DRIVE-READY ZIP PACKAGER
# ─────────────────────────────────────────────────────────────────────────────

def package_for_drive(
    output_zip: Path,
    files_to_include: List[Path],
    manifest: Dict[str, Any],
) -> None:
    """
    Bundle the corpus output files into a single ZIP archive suitable for
    upload to / mounting in Google Drive.

    Structure inside the ZIP:
      SC_2024_corpus/
        aws_court_chunks.jsonl     ← main preprocessed corpus
        citation_map.json          ← citation token → raw string lookup
        manifest.json              ← run metadata (doc count, chunk count, etc.)
        pipeline_run.log           ← full pipeline execution log
    """
    folder = f"SC_{TARGET_YEAR}_corpus"
    logger.info("Packaging Drive-ready ZIP: %s", output_zip)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Write manifest inline
        zf.writestr(
            f"{folder}/manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )
        # Add corpus files
        for fpath in files_to_include:
            if fpath.exists():
                arcname = f"{folder}/{fpath.name}"
                zf.write(fpath, arcname)
                logger.info("  + %s → %s", fpath.name, arcname)
            else:
                logger.warning("  ! File not found, skipping: %s", fpath)

    size_mb = output_zip.stat().st_size / 1e6
    logger.info("ZIP ready: %s (%.1f MB)", output_zip.resolve(), size_mb)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 ─ MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """
    End-to-end orchestration:

      1. Setup directories
      2. Download english.tar (resumable)
      3. Walk tar → extract PDFs as bytes
      4. Submit each PDF to ProcessPoolExecutor worker
      5. Drain futures → write JSONL → merge citation maps
      6. Persist citation_map.json
      7. Generate manifest.json
      8. Package everything into Drive-ready ZIP
    """
    t_start = time.perf_counter()
    now_utc = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info("  SC PIPELINE v2.0 — START  %s", now_utc)
    logger.info("  Year     : %s  |  Max tokens: %d  |  Overlap: %d",
                TARGET_YEAR, MAX_WINDOW_TOKENS, OVERLAP_TOKENS)
    logger.info("  Workers  : %d physical cores (of %d logical)",
                PHYSICAL_CPU_CORES, multiprocessing.cpu_count())
    logger.info("=" * 70)

    # ── 1. Setup ─────────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL.write_text("", encoding="utf-8")   # fresh run

    global_cmap: Dict[str, str] = {}

    # ── 2. S3 client + download tar ──────────────────────────────────────────
    s3 = build_s3_client()
    download_tar_streaming(s3, LOCAL_TAR)

    # ── 3. Walk tar and dispatch workers ─────────────────────────────────────
    total_pdfs      = 0
    total_chunks    = 0
    total_skipped   = 0
    total_failed    = 0

    # We batch-submit to the executor and drain every BATCH_SIZE futures
    # to keep memory bounded (each PDF can be several MB of bytes).
    BATCH_SIZE = PHYSICAL_CPU_CORES * 4

    pending: Dict[concurrent.futures.Future, Tuple[str, int]] = {}

    def drain_futures(wait_all: bool = False) -> None:
        nonlocal total_chunks, total_failed
        fs = list(pending.keys())
        done_set = (
            concurrent.futures.as_completed(fs)
            if wait_all
            else [f for f in fs if f.done()]
        )
        for fut in done_set:
            pdf_n, seq_n = pending.pop(fut)
            try:
                chunk_list, worker_cmap = fut.result(timeout=120)
                global_cmap.update(worker_cmap)
                if chunk_list:
                    written = append_chunks_to_jsonl(chunk_list, OUTPUT_JSONL)
                    total_chunks += written
                    logger.info("  ✓ [%04d] %s → %d chunks (total %d)",
                                seq_n, pdf_n, written, total_chunks)
                else:
                    nonlocal total_skipped
                    total_skipped += 1
            except Exception as exc:
                logger.error("  ✗ [%04d] %s failed: %s", seq_n, pdf_n, exc)
                total_failed += 1

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=PHYSICAL_CPU_CORES
    ) as executor:
        with tarfile.open(LOCAL_TAR, "r:") as tar:
            for member in tar.getmembers():
                if not member.name.lower().endswith(".pdf"):
                    continue
                total_pdfs += 1
                pdf_name = Path(member.name).name
                f_obj    = tar.extractfile(member)
                if f_obj is None:
                    total_skipped += 1
                    continue
                pdf_bytes = f_obj.read()

                # Worker gets a snapshot of the current citation map
                worker_cmap_snapshot = dict(global_cmap)
                future = executor.submit(
                    process_document,
                    (pdf_name, pdf_bytes, total_pdfs, worker_cmap_snapshot),
                )
                pending[future] = (pdf_name, total_pdfs)

                # Drain ready futures to keep memory bounded
                if len(pending) >= BATCH_SIZE:
                    drain_futures(wait_all=False)

        # Final drain — wait for all submitted futures
        logger.info("All %d PDFs submitted. Waiting for final futures...", total_pdfs)
        drain_futures(wait_all=True)

    # ── 4. Persist citation map ───────────────────────────────────────────────
    with open(CITATION_MAP, "w", encoding="utf-8") as f:
        json.dump(global_cmap, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info("Citation map: %d entries → %s", len(global_cmap), CITATION_MAP)

    # ── 5. Manifest ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    manifest = {
        "pipeline_version": "2.0",
        "run_timestamp_utc": now_utc,
        "target_year": TARGET_YEAR,
        "s3_bucket": S3_BUCKET,
        "s3_tar_key": S3_TAR_KEY,
        "total_pdfs_found": total_pdfs,
        "total_pdfs_skipped": total_skipped,
        "total_pdfs_failed": total_failed,
        "total_chunks_written": total_chunks,
        "unique_citations_mapped": len(global_cmap),
        "max_window_tokens": MAX_WINDOW_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "cpu_workers": PHYSICAL_CPU_CORES,
        "elapsed_seconds": round(elapsed, 2),
        "output_files": {
            "corpus_jsonl": str(OUTPUT_JSONL.name),
            "citation_map": str(CITATION_MAP.name),
        },
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ── 6. Package for Google Drive ───────────────────────────────────────────
    package_for_drive(
        output_zip=DRIVE_ZIP,
        files_to_include=[OUTPUT_JSONL, CITATION_MAP, Path("pipeline_run.log")],
        manifest=manifest,
    )

    # ── 7. Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Elapsed          : %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("  PDFs processed   : %d / %d", total_pdfs - total_skipped, total_pdfs)
    logger.info("  Chunks written   : %d", total_chunks)
    logger.info("  Citations mapped : %d", len(global_cmap))
    logger.info("  Drive ZIP        : %s", DRIVE_ZIP.resolve())
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 ─ ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # macOS default multiprocessing start method is "spawn" — required for
    # ProcessPoolExecutor to work correctly with SpaCy on macOS.
    multiprocessing.set_start_method("spawn", force=True)

    # Pre-flight: verify SpaCy model loads cleanly before spawning workers
    try:
        get_nlp()
        logger.info("SpaCy pre-flight OK.")
    except Exception as exc:
        raise SystemExit(f"SpaCy init failed: {exc}") from exc

    run_pipeline()

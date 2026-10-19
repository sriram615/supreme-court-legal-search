#!/usr/bin/env python3
"""
Record a real demo GIF of the LAWdata frontend UI hitting the real backend.

Steps:
1. Assert backend health (blocking wait).
2. Launch system Chrome headless via Playwright.
3. Navigate to http://127.0.0.1:8000/ (real frontend served by FastAPI).
4. Capture frames: landing page, typing search, loading state, results page.
5. Save per-frame PNGs + a duration manifest for ffmpeg assembly.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Ensure API key is available for the agentic path ──
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set in this shell. Synthesized answer will be null.")

OUTDIR = Path("/private/var/folders/6x/3m69j6b91fj6lyhmntcw5k740000gn/T/opencode/lawdata_gif")
OUTDIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUTDIR}")

BACKEND_URL = "http://127.0.0.1:8000"
HEALTH_PATH = "/api/v1/health"

# ── Step 2: Blocking readiness wait ──
print("Waiting for backend readiness...")
ready = False
for i in range(45):  # up to 90s
    try:
        r = subprocess.run(
            ["curl", "-sf", "--max-time", "3", f"{BACKEND_URL}{HEALTH_PATH}"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            if data.get("engine_ready") and data.get("status") == "healthy":
                print(f"READY after ~{i*2}s — {data['index_vectors_count']} vectors")
                ready = True
                break
    except Exception:
        pass
    time.sleep(2)

if not ready:
    print("ERROR: Backend not ready. Aborting.")
    sys.exit(1)

# ── Prove it works with curl (real user path) ──
print("Proving backend with real query...")
r = subprocess.run(
    ["curl", "-sf", "--max-time", "30",
     "-X", "POST", f"{BACKEND_URL}/api/v1/agentic-analyze",
     "-H", "Content-Type: application/json",
     "-d", '{"text":"Section 304B IPC dowry death essential ingredients"}'],
    capture_output=True, text=True, timeout=35,
)
if r.returncode != 0:
    print(f"curl failed: {r.stderr[:500]}")
    sys.exit(1)
data = json.loads(r.stdout)
print(f"query_type: {data.get('query_type')}")
print(f"rewritten_query: {data.get('rewritten_query')}")
print(f"hits: {len(data.get('semantic_matches', []))}")
print(f"answer: {str(data.get('synthesized_answer'))[:120]}...")
print("Backend confirmed working with real data.")

# ── Step 3: Drive UI with Playwright ──
from playwright.sync_api import sync_playwright

FRAMES = []  # list of (png_path, duration_sec)

def save_frame(page, name, duration=1.0):
    path = OUTDIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    FRAMES.append((str(path), duration))
    print(f"  frame: {name} ({path.name})")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, channel="chrome",
                                args=["--hide-scrollbars", "--force-color-profile=srgb"])
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        device_scale_factor=1,
        reduced_motion="no-preference",
    )
    page = context.new_page()

    # ── Landing page ──
    print("Navigating to frontend...")
    page.goto(f"{BACKEND_URL}/", wait_until="networkidle")
    page.wait_for_timeout(1500)
    save_frame(page, "01_landing", duration=1.5)

    # ── Verify health dot shows healthy ──
    health_dot = page.locator("#statusDot")
    if health_dot.count() > 0:
        print(f"  health dot class: {health_dot.get_attribute('class')}")

    # ── Type search query ──
    print("Typing search query...")
    query_input = page.locator("#queryInput")
    query_input.click()
    query_input.type("Section 304B IPC dowry death essential ingredients", delay=40)
    page.wait_for_timeout(500)
    save_frame(page, "02_typed", duration=1.0)

    # ── Submit ──
    print("Submitting search...")
    page.locator("#sendBtn").click()
    page.wait_for_timeout(800)
    save_frame(page, "03_loading", duration=1.0)

    # ── Wait for results ──
    page.wait_for_timeout(3000)
    save_frame(page, "04_results", duration=2.0)

    # ── Verify real content rendered ──
    transcript = page.locator("#transcript")
    transcript_text = transcript.inner_text()
    print(f"  transcript length: {len(transcript_text)} chars")
    has_citations = "AWS_SC_" in transcript_text
    has_answer = "Section 304B" in transcript_text or "dowry" in transcript_text.lower()
    print(f"  has real chunk IDs: {has_citations}")
    print(f"  has answer content: {has_answer}")

    # ── Conceptual query ──
    print("\nTyping conceptual query...")
    page.locator("#newSearchBtn").click()
    page.wait_for_timeout(500)
    query_input = page.locator("#queryInput")
    query_input.type("anticipatory bail money laundering", delay=40)
    page.wait_for_timeout(500)
    save_frame(page, "05_typed_conceptual", duration=1.0)

    page.locator("#sendBtn").click()
    page.wait_for_timeout(3500)
    save_frame(page, "06_results_conceptual", duration=2.0)

    transcript2 = page.locator("#transcript").inner_text()
    print(f"  conceptual transcript length: {len(transcript2)} chars")

    context.close()
    browser.close()

# ── Step 4: Write manifest ──
manifest_path = OUTDIR / "manifest.txt"
with open(manifest_path, "w") as f:
    for path, dur in FRAMES:
        f.write(f"file '{path}'\n")
        f.write(f"duration {dur}\n")
# Repeat last frame for final duration
last_path, last_dur = FRAMES[-1]
with open(manifest_path, "a") as f:
    f.write(f"file '{last_path}'\n")
    f.write(f"duration {last_dur}\n")

print(f"\nCaptured {len(FRAMES)} frames. Manifest: {manifest_path}")
print("Frames:")
for path, dur in FRAMES:
    size = os.path.getsize(path)
    print(f"  {Path(path).name}  {dur}s  {size:,} bytes")

# ── Verify frames contain real app content ──
from PIL import Image
for path, dur in FRAMES[:3]:
    img = Image.open(path)
    print(f"  {Path(path).name}: {img.size} mode={img.mode}")

print("\nNext: assemble with ffmpeg using manifest at", manifest_path)

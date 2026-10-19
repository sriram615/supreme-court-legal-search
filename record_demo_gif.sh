#!/bin/bash
# record_demo_gif.sh — Capture a screen region to a GIF for portfolio demo.
#
# Prerequisites (already on your Mac):
#   ffmpeg   (brew install ffmpeg)
#   screencapture (built into macOS)
#
# Usage:
#   1. Start the server:   uvicorn app:app --host 127.0.0.1 --port 8000
#   2. Open the console:   http://127.0.0.1:8000/  (in Chrome/Safari)
#   3. Resize browser window to ~800x600 and position it where you want.
#   4. Run this script:    ./record_demo_gif.sh demo.gif 15
#   5. While it records, perform your search (e.g. "Section 304B IPC dowry death").
#   6. After the duration ends, ffmpeg compiles the GIF automatically.
#
# Output: demo.gif in the current directory.

set -euo pipefail

OUTFILE="${1:-demo.gif}"
DURATION="${2:-15}"

# Try to auto-detect the avfoundation screen device
SCREEN_DEV=$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -E "Capture screen [0-9]" | head -1 | grep -oE "Capture screen [0-9]+" | grep -oE "[0-9]+")

if [ -z "$SCREEN_DEV" ]; then
    echo "Could not auto-detect screen device. Listing devices:"
    ffmpeg -f avfoundation -list_devices true -i "" 2>&1
    echo ""
    echo "Manually set SCREEN_DEV in this script (e.g. SCREEN_DEV=1)."
    exit 1
fi

echo "Recording screen device $SCREEN_DEV for $DURATION seconds..."
echo "Perform your search now. GIF will be saved to: $OUTFILE"
echo ""

ffmpeg -f avfoundation -i "${SCREEN_DEV}:0" \
    -t "$DURATION" \
    -vf "fps=10,scale=720:-1:flags=lanczos" \
    -gifflags +transdiff \
    -y "$OUTFILE"

echo ""
echo "Done: $OUTFILE"
echo "File size: $(du -h "$OUTFILE" | cut -f1)"

#!/usr/bin/env bash
#
# Decode a trained ContactUBS ffhq_psp bridge into images: takes the
# predicted latent trajectory (x_pred.npy, plus its starting point x0.npy)
# and runs it through the pSp decoder, producing a grid of 10 sample
# identities transitioning from children to adults across the trajectory's
# timesteps (rows = timesteps, columns = identities).
#
# Wraps bridge_visualizer_psp.py: copies x0.npy/x_pred.npy from a ContactUBS
# run (written by `main.py` to its own working directory) into
# contactubs_images_psp/, the convention that script expects, then runs it.
#
# Usage:
#   ./decode_bridge_to_images.sh [x0.npy] [x_pred.npy]
#     defaults: ../../ContactUBS/x0.npy and ../../ContactUBS/x_pred.npy
#     (i.e. whatever the most recent `main.py --params ffhq_psp*` run wrote)
#
# Output: contactubs_images_psp/transition_grid_psp.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

X0_SRC="${1:-../../ContactUBS/x0.npy}"
XPRED_SRC="${2:-../../ContactUBS/x_pred.npy}"
OUT_DIR="contactubs_images_psp"

if [[ ! -f "$X0_SRC" ]]; then
    echo "error: $X0_SRC not found" >&2
    exit 1
fi
if [[ ! -f "$XPRED_SRC" ]]; then
    echo "error: $XPRED_SRC not found" >&2
    exit 1
fi

echo "x0:     $X0_SRC (modified $(date -r "$X0_SRC" '+%Y-%m-%d %H:%M:%S'))"
echo "x_pred: $XPRED_SRC (modified $(date -r "$XPRED_SRC" '+%Y-%m-%d %H:%M:%S'))"

mkdir -p "$OUT_DIR"
cp "$X0_SRC" "$OUT_DIR/x0.npy"
cp "$XPRED_SRC" "$OUT_DIR/x_pred.npy"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --project ../../ContactUBS --with setuptools python bridge_visualizer_psp.py

echo
echo "Done: $OUT_DIR/transition_grid_psp.png"

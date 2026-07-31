#!/usr/bin/env bash
#
# Computes FID (in pSp-decoded pixel space) for the ContactUBS FFHQPSPDataset
# ablation, using compute_fid_psp.py. Follows the same `uv run --project
# ../../ContactUBS --with ...` convention as check_psp_requirements.sh /
# run_psp_pipeline.sh: pytorch-fid and setuptools are layered on top per-
# invocation (setuptools because pixel2style2pixel's stylegan2 ops
# unconditionally import torch.utils.cpp_extension), without touching
# ContactUBS's pyproject.toml/uv.lock.
#
# Usage:
#   ./compute_fid_psp.sh
#       No args: auto-discovers every x_pred*.npy in results dir
#       (default: contactubs_images_psp) and reports FID for each
#       (e.g. x_pred.npy and x_pred_1.npy), saving JSON per run.
#
#   ./compute_fid_psp.sh --tag _1 --num-real 5000
#       Any arg forwards straight to compute_fid_psp.py for a single run
#       (see `python compute_fid_psp.py --help` for all options).
#
# Env overrides:
#   UV_PROJECT=../../ContactUBS ./compute_fid_psp.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UV_PROJECT="${UV_PROJECT:-../../ContactUBS}"
PY_RUN=(uv run --project "$UV_PROJECT" --with pytorch-fid --with setuptools python)

RESULTS_DIR="contactubs_images_psp"
OUTPUT_DIR="fid_results"
mkdir -p "$OUTPUT_DIR"

echo "== Checking environment (uv project: $UV_PROJECT) =="
"${PY_RUN[@]}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "error: no CUDA GPU visible inside the uv env (pSp decoder + InceptionV3 need one)" >&2
    exit 1
}

if [[ $# -gt 0 ]]; then
    "${PY_RUN[@]}" compute_fid_psp.py --output "$OUTPUT_DIR/fid_manual.json" "$@"
    exit 0
fi

shopt -s nullglob
found=0
for xpred in "$RESULTS_DIR"/x_pred*.npy; do
    base="$(basename "$xpred" .npy)"
    tag="${base#x_pred}"          # "" or "_1", "_2", ...
    x0file="$RESULTS_DIR/x0${tag}.npy"
    if [[ ! -f "$x0file" ]]; then
        echo "skip: ${x0file} not found for ${xpred}"
        continue
    fi
    found=1
    echo
    echo "== FID for ${RESULTS_DIR} (tag='${tag}') =="
    "${PY_RUN[@]}" compute_fid_psp.py \
        --results-dir "$RESULTS_DIR" --tag "$tag" \
        --output "$OUTPUT_DIR/fid${tag}.json"
done

if [[ $found -eq 0 ]]; then
    echo "No x_pred*.npy files found in $RESULTS_DIR" >&2
    exit 1
fi

echo
echo "All results saved under $OUTPUT_DIR/"

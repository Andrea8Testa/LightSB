#!/usr/bin/env bash
#
# ContactUBS FFHQ pSp data pipeline
# =================================
#
# Rebuilds the FFHQ children->adult dataset in the pixel2style2pixel (pSp)
# W+ latent space, so ContactUBS/params/ffhq_psp.py can be trained the same
# way params/ffhq.py is trained on the ALAE latent space.
#
# Steps:
#   1. Download the ALAE latents + age/gender labels     (download_data.py)
#   2. Download the ALAE FFHQ decoder checkpoint          (gdown, S3 fallback)
#   3. Download the pretrained pSp FFHQ encoder checkpoint (gdown)
#   4. Decode every ALAE latent -> 256x256 face image     (decode_alae_to_images.py)
#   5. Re-encode every face image -> pSp W+ latent         (encode_psp_latents.py)
#
# Output: data/psp_latents.npy, shape (N, 18, 512), float32. It is aligned
# index-for-index with data/age.npy / data/gender.npy (both produced by
# step 1), which is exactly what dataset.py's FFHQPSPDataset expects.
#
# Every step is run through `uv run` against the ContactUBS project (same
# torch/numpy/tqdm/matplotlib env used for training). ALAE and the
# downloaders additionally need yacs/gdown/requests, and pixel2style2pixel's
# stylegan2 op module unconditionally imports torch.utils.cpp_extension
# (needs setuptools, not bundled by default in modern venvs) even though the
# native kernel it tries to JIT-compile is optional (falls back to a pure
# PyTorch implementation) - none of these are ContactUBS dependencies, so
# they're layered on top per-invocation via `uv run --with`, without
# touching ContactUBS's pyproject.toml/uv.lock.
#
# Usage:
#   ./run_psp_pipeline.sh              # full run
#   ./run_psp_pipeline.sh --limit 200  # smoke test on the first 200 latents
#
# Env overrides:
#   UV_PROJECT=../../ContactUBS ./run_psp_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UV_PROJECT="${UV_PROJECT:-../../ContactUBS}"
EXTRA_DEPS=(--with yacs --with gdown --with requests --with setuptools)
PY_RUN=(uv run --project "$UV_PROJECT" "${EXTRA_DEPS[@]}" python)

LIMIT_ARGS=()
if [[ "${1:-}" == "--limit" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "error: --limit requires a value" >&2
        exit 1
    fi
    LIMIT_ARGS=(--limit "$2")
fi

# A file that "exists" but was left behind by a job killed mid-np.save (OOM,
# walltime) is worse than a missing one: it looks done and silently poisons
# the next stage. mmap_mode='r' fails fast on a truncated file without
# reading the whole array back in.
npy_ok() {
    "${PY_RUN[@]}" -c "
import sys
import numpy as np
try:
    np.load(sys.argv[1], mmap_mode='r').shape
except Exception:
    sys.exit(1)
" "$1" >/dev/null 2>&1
}

echo "== [0/5] Checking environment (uv project: $UV_PROJECT) =="
"${PY_RUN[@]}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "error: no CUDA GPU visible inside the uv env (both ALAE decoding and pSp encoding require one)" >&2
    exit 1
}

echo "== [1/5] Downloading ALAE latents + age/gender labels =="
mkdir -p data
if npy_ok data/latents.npy && npy_ok data/age.npy && npy_ok data/gender.npy && npy_ok data/test_images.npy; then
    echo "data/{latents,age,gender,test_images}.npy already present and valid, skipping."
else
    "${PY_RUN[@]}" download_data.py
fi

echo "== [2/5] Downloading ALAE FFHQ decoder checkpoint =="
mkdir -p ALAE/training_artifacts/ffhq
# last_checkpoint pins the exact file the loader reads (model_157.pth)
if [[ -f ALAE/training_artifacts/ffhq/model_157.pth ]]; then
    echo "ALAE FFHQ checkpoint already present, skipping."
else
    "${PY_RUN[@]}" - <<'PYEOF'
import os
import gdown
import requests

def download_from_gdrive(file_id, directory):
    os.makedirs(directory, exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output=directory, quiet=False)

def download_from_url(url, directory):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, url.split("/")[-1])
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

directory = "ALAE/training_artifacts/ffhq"
try:
    download_from_gdrive("170Qldnn28IwnVm9CQEq1AZhVsK7PJ0Xz", directory)
    download_from_gdrive("1QESywJW8N-g3n0Csy0clztuJV99g8pRm", directory)
    download_from_gdrive("18BzFYKS3icFd1DQKKTeje7CKbEKXPVug", directory)
except Exception:
    download_from_url("https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_submitted.pth", directory)
    download_from_url("https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_194.pth", directory)
    download_from_url("https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth", directory)
PYEOF
fi

echo "== [3/5] Downloading pretrained pSp FFHQ encoder checkpoint =="
mkdir -p ../pixel2style2pixel/pretrained_models
PSP_CKPT="../pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt"
if [[ -f "$PSP_CKPT" ]]; then
    echo "pSp FFHQ encoder checkpoint already present, skipping."
else
    "${PY_RUN[@]}" -c "
import sys, gdown
gdown.download('https://drive.google.com/uc?id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0', sys.argv[1], quiet=False)
" "$PSP_CKPT"
fi

echo "== [4/5] Decoding ALAE latents -> 256x256 face images (ALAE decoder) =="
if [[ ${#LIMIT_ARGS[@]} -eq 0 ]] && npy_ok data/psp_input_images_256.npy; then
    echo "data/psp_input_images_256.npy already present and valid, skipping."
else
    "${PY_RUN[@]}" decode_alae_to_images.py "${LIMIT_ARGS[@]}"
fi

echo "== [5/5] Encoding face images -> pSp W+ latents (pSp encoder) =="
if [[ ${#LIMIT_ARGS[@]} -eq 0 ]] && npy_ok data/psp_latents.npy; then
    echo "data/psp_latents.npy already present and valid, skipping."
else
    "${PY_RUN[@]}" encode_psp_latents.py
fi

echo
echo "Done. data/psp_latents.npy is ready, aligned with data/age.npy / data/gender.npy."
echo "Train ContactUBS on it with:"
echo "  (cd ../../ContactUBS && uv run python main.py --params ffhq_psp)"

"""
Stage 1 of the ALAE -> pSp latent conversion pipeline.

Decodes every ALAE latent in data/latents.npy into its corresponding face
image via the ALAE FFHQ decoder, downsamples to 256x256 (the resolution the
pixel2style2pixel encoder expects), and stores the result as a single uint8
array. Stage 2 (encode_psp_latents.py) consumes that array to produce the
pSp W+ latents.

Run from the LightSB/ directory:
    python decode_alae_to_images.py
"""
import os
import sys
import argparse

sys.path.append("ALAE")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from alae_ffhq_inference import load_model, decode

parser = argparse.ArgumentParser()
parser.add_argument("--latents", type=str, default="data/latents.npy")
parser.add_argument("--out", type=str, default="data/psp_input_images_256.npy")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--limit", type=int, default=None,
                     help="Only decode the first N latents (for smoke testing).")
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("This script expects a CUDA GPU (ALAE decoder is heavy on CPU).")
device = torch.device("cuda")

print(f"Loading ALAE FFHQ model...")
model = load_model("ALAE/configs/ffhq.yaml", training_artifacts_dir="ALAE/training_artifacts/ffhq/")
model = model.to(device)
model.eval()

latents = np.load(args.latents)
if args.limit is not None:
    latents = latents[:args.limit]
n = latents.shape[0]
print(f"Decoding {n} ALAE latents ({latents.shape[1]}-d) -> {args.image_size}x{args.image_size} images")

out_images = np.empty((n, args.image_size, args.image_size, 3), dtype=np.uint8)

with torch.no_grad():
    for i in tqdm(range(0, n, args.batch_size)):
        batch = latents[i:i + args.batch_size]
        z = torch.tensor(batch, dtype=torch.float32, device=device)

        img = decode(model, z)  # (B, 3, 1024, 1024), float in roughly [-1, 1]
        img = F.interpolate(img, size=(args.image_size, args.image_size),
                             mode="bilinear", align_corners=False)
        img = ((img * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
        img = img.permute(0, 2, 3, 1).cpu().numpy()

        out_images[i:i + img.shape[0]] = img

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
tmp_out = args.out + ".tmp.npy"  # np.save always appends .npy; keep it on the tmp name too
np.save(tmp_out, out_images)
os.replace(tmp_out, args.out)  # atomic: never leaves a truncated file at args.out
print(f"Saved {out_images.shape} uint8 array to {args.out}")

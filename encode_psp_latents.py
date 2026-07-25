"""
Stage 2 of the ALAE -> pSp latent conversion pipeline.

Takes the 256x256 face images produced by decode_alae_to_images.py (which
were themselves decoded from the ALAE latents in data/latents.npy) and runs
them through the pretrained pixel2style2pixel encoder to obtain the
corresponding pSp W+ latents (one 512-d StyleGAN style code per layer, 18
layers for the 1024x1024 FFHQ generator).

These pSp latents are aligned index-for-index with data/latents.npy (and
therefore with data/age.npy / data/gender.npy), so ContactUBS's
FFHQPSPDataset can reuse the same age/gender masks.

Run from the LightSB/ directory:
    python encode_psp_latents.py
"""
import os
import sys
import argparse
from argparse import Namespace

sys.path.append(os.path.join("..", "pixel2style2pixel"))

import numpy as np
import torch
from tqdm import tqdm

from models.psp import pSp

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, default="data/psp_input_images_256.npy")
parser.add_argument("--out", type=str, default="data/psp_latents.npy")
parser.add_argument("--checkpoint", type=str,
                     default="../pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt")
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("This script expects a CUDA GPU.")
device = torch.device("cuda")

print(f"Loading pSp encoder from {args.checkpoint} ...")
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
opts = ckpt["opts"]
opts["checkpoint_path"] = args.checkpoint
opts.setdefault("output_size", 1024)
opts["device"] = str(device)
opts = Namespace(**opts)

net = pSp(opts)
net.eval().to(device)
print(f"pSp encoder loaded. n_styles = {opts.n_styles}")

images = np.load(args.images)
n = images.shape[0]
print(f"Encoding {n} images ({images.shape[1]}x{images.shape[2]}) -> pSp W+ latents")

out_latents = np.empty((n, opts.n_styles, 512), dtype=np.float32)

with torch.no_grad():
    for i in tqdm(range(0, n, args.batch_size)):
        batch = images[i:i + args.batch_size]
        x = torch.tensor(batch, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        x = x / 255.0
        x = (x - 0.5) / 0.5  # match pSp's training normalization: mean=std=0.5

        codes = net.encoder(x)
        if opts.start_from_latent_avg:
            if opts.learn_in_w:
                codes = codes + net.latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)

        out_latents[i:i + codes.shape[0]] = codes.cpu().numpy()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
tmp_out = args.out + ".tmp.npy"  # np.save always appends .npy; keep it on the tmp name too
np.save(tmp_out, out_latents)
os.replace(tmp_out, args.out)  # atomic: never leaves a truncated file at args.out
print(f"Saved {out_latents.shape} float32 array to {args.out}")

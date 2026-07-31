"""
FID (Frechet Inception Distance) for the ContactUBS FFHQPSPDataset ablation, in
pSp-decoded ambient/pixel space.

Reads x0[<tag>].npy / x_pred[<tag>].npy (produced by ContactUBS/main.py with
--params ffhq_psp, then copied here -- see bridge_visualizer_psp.py), decodes
the final bridge timestep x_pred[-1] and a random sample of real target-class
pSp latents through the pretrained pSp decoder, and reports FID between them.
A baseline FID(x0 vs real target) is also reported so the improvement from the
bridge is visible.

Note: this duplicates ffhq_utils.py's get_activations()/calculate_fid() rather
than importing them. Those two functions are decoder-agnostic (they only see
uint8 images), but ffhq_utils.py unconditionally does
`from alae_ffhq_inference import load_model, encode, decode`, which as an
import-time side effect calls `torch.set_default_device("cuda")` globally and
pulls in ALAE-only dependencies (yacs, ALAE's net/model/checkpointer modules).
None of that is needed for pSp decoding, so it's not imported here.
ffhq_utils.decode_latents()/load_data() are ALAE-specific in a way that isn't
reusable at all for pSp: they call ALAE's decode(model, z) on flat 512-d
latents, while pSp needs its own model.decoder([codes], input_is_latent=True,
...) on (18, 512) W+ codes, and load_data() reads data/latents.npy (ALAE,
512-d) rather than data/psp_latents.npy (pSp, 18x512).

Run from the LightSB/ directory:
    python compute_fid_psp.py --results-dir contactubs_images_psp
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join("..", "pixel2style2pixel"))
from models.psp import pSp

from pytorch_fid.inception import InceptionV3
from scipy.linalg import sqrtm

N_STYLES = 18
STYLE_DIM = 512


def get_activations(images, model, device, batch_size=50):
    model.eval()
    activations = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            batch = torch.tensor(batch).float() / 255.0
            batch = batch.permute(0, 3, 1, 2).to(device)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            pred = model(batch)[0]
            pred = pred.squeeze(3).squeeze(2)
            activations.append(pred.cpu().numpy())
    return np.concatenate(activations, axis=0)


def calculate_fid(real_images, fake_images, device):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)

    act1 = get_activations(real_images, inception, device)
    act2 = get_activations(fake_images, inception, device)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def get_ffhq_mask(gender, age, category):
    """Mirrors ContactUBS/dataset.py::_get_ffhq_mask exactly."""
    if category == "MAN":
        return (gender == "male").reshape(-1)
    elif category == "WOMAN":
        return (gender == "female").reshape(-1)
    elif category == "ADULT":
        return ((age >= 18) & (age != -1)).reshape(-1)
    elif category == "CHILDREN":
        return ((age < 18) & (age != -1)).reshape(-1)
    raise ValueError(f"Unknown category: {category}")


def load_psp_decoder(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    opts = ckpt["opts"]
    opts["checkpoint_path"] = checkpoint_path
    opts.setdefault("output_size", 1024)
    opts["device"] = str(device)
    opts = argparse.Namespace(**opts)

    model = pSp(opts)
    model = model.to(device)
    model.eval()
    return model


def decode_psp_latents(model, latents_flat, device, batch_size=16):
    """latents_flat: (N, 9216) raw (unstandardized) pSp W+ codes -> uint8 images (N, H, W, 3)."""
    imgs = []
    with torch.no_grad():
        for i in tqdm(range(0, latents_flat.shape[0], batch_size), desc="pSp decode"):
            batch = latents_flat[i:i + batch_size]
            codes = torch.tensor(batch, dtype=torch.float32, device=device).view(-1, N_STYLES, STYLE_DIM)
            img, _ = model.decoder([codes], input_is_latent=True,
                                    randomize_noise=False, return_latents=True)
            img = ((img * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
            img = img.permute(0, 2, 3, 1).cpu().numpy()
            imgs.append(img)
    return np.concatenate(imgs, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="contactubs_images_psp",
                         help="Directory containing x0<tag>.npy / x_pred<tag>.npy")
    parser.add_argument("--tag", default="",
                         help='Suffix for a specific run, e.g. "_1" -> x0_1.npy/x_pred_1.npy')
    parser.add_argument("--target-data", default="ADULT",
                         choices=["MAN", "WOMAN", "ADULT", "CHILDREN"],
                         help="Real-image class to compare the bridge output against.")
    parser.add_argument("--input-data", default="CHILDREN",
                         choices=["MAN", "WOMAN", "ADULT", "CHILDREN"],
                         help="Class of the x0 source samples (only used to label the baseline FID).")
    parser.add_argument("--num-real", type=int, default=2000,
                         help="How many real target-class images to sample as the FID reference set.")
    parser.add_argument("--decode-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-baseline", action="store_true",
                         help="Skip the FID(x0 vs real target) baseline, only report FID(x_pred[-1] vs real target).")
    parser.add_argument("--psp-checkpoint",
                         default="../pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=None, help="Optional path to write results as JSON.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script expects a CUDA GPU (pSp decoder + InceptionV3 are heavy on CPU).")
    device = torch.device("cuda")
    np.random.seed(args.seed)

    x0_path = os.path.join(args.results_dir, f"x0{args.tag}.npy")
    xpred_path = os.path.join(args.results_dir, f"x_pred{args.tag}.npy")
    x0 = np.load(x0_path)
    x_pred = np.load(xpred_path)
    print(f"x0: {x0.shape}, x_pred: {x_pred.shape}")

    # x0 has an appended homogeneous/mass column (see ContactUBS/main.py: x0 =
    # cat([x0_wo_mass, ones]) ); x_pred does not (results[..., :-1] already
    # dropped it).
    x0_latents_std = x0[:, :-1]

    psp_latents_raw = np.load(os.path.join(args.data_dir, "psp_latents.npy")).reshape(-1, N_STYLES * STYLE_DIM)
    psp_mean = psp_latents_raw.mean(axis=0)
    psp_std = np.maximum(psp_latents_raw.std(axis=0), 1e-3)

    gender = np.load(os.path.join(args.data_dir, "gender.npy"))
    age = np.load(os.path.join(args.data_dir, "age.npy"))
    target_mask = get_ffhq_mask(gender, age, args.target_data)
    target_pool = psp_latents_raw[target_mask]
    if args.num_real > target_pool.shape[0]:
        raise ValueError(
            f"--num-real {args.num_real} exceeds available {args.target_data} pool size {target_pool.shape[0]}"
        )
    real_idx = np.random.choice(target_pool.shape[0], size=args.num_real, replace=False)
    real_latents = target_pool[real_idx]

    print("Loading pSp decoder...")
    model = load_psp_decoder(args.psp_checkpoint, device)

    print(f"Decoding {args.num_real} real {args.target_data} images...")
    real_images = decode_psp_latents(model, real_latents, device, batch_size=args.decode_batch_size)

    fake_latents_std = x_pred[-1]  # (N, 9216): final-timestep bridge prediction, standardized space
    fake_latents = fake_latents_std * psp_std + psp_mean
    if fake_latents.shape[0] < 2048:
        print(f"warning: only {fake_latents.shape[0]} fake samples -- FID's covariance estimate "
              f"(2048-d Inception features) will be rank-deficient and noisy at this sample size.")
    print(f"Decoding {fake_latents.shape[0]} bridge-predicted (x_pred[-1]) images...")
    fake_images = decode_psp_latents(model, fake_latents, device, batch_size=args.decode_batch_size)

    print("Computing FID(x_pred[-1] vs real target)...")
    fid_final = calculate_fid(real_images, fake_images, device)

    results = {
        "results_dir": args.results_dir,
        "tag": args.tag,
        "target_data": args.target_data,
        "num_real": args.num_real,
        "num_fake": int(fake_images.shape[0]),
        "fid_final": fid_final,
    }

    if not args.skip_baseline:
        x0_latents_raw = x0_latents_std * psp_std + psp_mean
        print(f"Decoding {x0_latents_raw.shape[0]} baseline ({args.input_data}, x0) source images...")
        baseline_images = decode_psp_latents(model, x0_latents_raw, device, batch_size=args.decode_batch_size)
        print("Computing FID(x0 vs real target) baseline...")
        fid_baseline = calculate_fid(real_images, baseline_images, device)
        results["input_data"] = args.input_data
        results["fid_baseline"] = fid_baseline

    print(json.dumps(results, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()

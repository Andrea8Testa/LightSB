"""
bridge_visualizer.py, adapted for the pixel2style2pixel (pSp) latent-space
ablation of the ContactUBS FFHQ experiment (params/ffhq_psp.py).

Expects x0.npy / x_pred.npy produced by `python main.py --params ffhq_psp
--model_name <run>` (copied into `directory_ref`, same convention as
bridge_visualizer.py). Those trajectories live in the flattened 9216-d
pSp W+ space; this script reshapes each 9216-d vector back to (18, 512)
and runs it through the pSp decoder (not the encoder) to get an image.
"""
import os, sys
sys.path.append(os.path.join("..", "pixel2style2pixel"))
import torch
import numpy as np
from argparse import Namespace
from tqdm import tqdm
from models.psp import pSp
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError

np.random.seed(seed=1)
directory_ref = "contactubs_images_psp"
print("directory_ref: ", directory_ref)

PSP_CHECKPOINT = "../pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt"
N_STYLES = 18
STYLE_DIM = 512

ckpt = torch.load(PSP_CHECKPOINT, map_location="cpu", weights_only=False)
opts = ckpt["opts"]
opts["checkpoint_path"] = PSP_CHECKPOINT
opts.setdefault("output_size", 1024)
opts["device"] = str(device)
opts = Namespace(**opts)

model = pSp(opts)
model = model.to(device)
model.eval()


def decode(model, codes_flat):
    """codes_flat: (B, 9216) -> images (B, 3, 1024, 1024) in [-1, 1]."""
    codes = codes_flat.view(-1, N_STYLES, STYLE_DIM)
    images, _ = model.decoder([codes], input_is_latent=True,
                               randomize_noise=False, return_latents=True)
    return images


x0_path = os.path.join(directory_ref, "x0.npy")
xpred_path = os.path.join(directory_ref, "x_pred.npy")

x0 = np.load(x0_path)
x_pred = np.load(xpred_path)

print("x0: ", x0.shape)
print("x_pred: ", x_pred.shape)

num_samples = 10
num_steps = 5

# choose random identities
inds = (0, 1, 2, 3, 4, 5, 6, 10, 11, 12)

# choose time indices
time_inds = np.linspace(0, x_pred.shape[0] - 1, num_steps).astype(int)

# select trajectories
selected_latents = x_pred[time_inds][:, inds]      # (steps, samples, 9216)
selected_latents = selected_latents.transpose(1, 0, 2)  # (samples, steps, 9216)

decoded_all = []

with torch.no_grad():
    for i in tqdm(range(num_samples)):
        row = []
        for k in range(num_steps):
            latent = torch.tensor(
                selected_latents[i, k],
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            img = decode(model, latent)
            img = (
                (img * 0.5 + 0.5) * 255
            ).clamp(0, 255).to(torch.uint8)
            img = img.cpu().permute(0, 2, 3, 1).numpy().squeeze(0)
            row.append(img)
        decoded_all.append(row)
decoded_all = np.array(decoded_all)

# --------------------------------------------------
# Plot grid
# --------------------------------------------------
fig, axes = plt.subplots(
    num_steps,
    num_samples,
    figsize=(num_samples * 2, num_steps * 2),
    dpi=200
)

for i in range(num_samples):
    for k in range(num_steps):
        axes[k, i].imshow(decoded_all[i, k])
        axes[k, i].axis("off")

plt.tight_layout(pad=0.05)
save_path = os.path.join(directory_ref, "transition_grid_psp.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved", save_path)

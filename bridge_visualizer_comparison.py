import os, sys
sys.path.append("ALAE")
import torch
import numpy as np
from tqdm import tqdm
from alae_ffhq_inference import load_model, encode, decode
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError

np.random.seed(seed=1)
directory_cfm = "conditional_flowmatching"
directory_cubs = "contactubs_images_atc"
directory_deep = "deep_images"
directory_var = "var_images"
model = load_model("ALAE/configs/ffhq.yaml", training_artifacts_dir="ALAE/training_artifacts/ffhq/")
model = model.to(device)
model.eval()

x0_path = os.path.join(directory_cfm, "x0.npy")
xpred_path_cfm = os.path.join(directory_cfm, "x_pred_atc.npy")
xpred_path_ubs = os.path.join(directory_cubs, "x_pred.npy")
xpred_path_deep = os.path.join(directory_deep, "x_pred_atc.npy")
xpred_path_var = os.path.join(directory_var, "x_pred_atc.npy")

inds = [0, 17, 41, 12, 4, 14, 39, 44, 30, 34]
x0 = np.load(xpred_path_cfm)[0][inds]
x_pred_cfm = np.load(xpred_path_cfm)[-1][inds]
x_pred_cubs = np.load(xpred_path_ubs)[-1][inds]
x_pred_deep = np.load(xpred_path_deep)[-1][inds]
x_pred_var = np.load(xpred_path_var)[-1][inds]

selected_latents = np.stack([
    x0,
    x_pred_cubs,
    x_pred_var,
    x_pred_deep,
    x_pred_cfm
], axis=0)

num_samples = len(inds)
num_rows = selected_latents.shape[0]
decoded_all = []

with torch.no_grad():
    for i in tqdm(range(num_samples)):
        row = []
        for k in range(num_rows):
            latent = torch.tensor(
                selected_latents[k, i],
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
print("decoded_all: ", decoded_all.shape)
# --------------------------------------------------
# Plot grid
# --------------------------------------------------
fig, axes = plt.subplots(
    num_rows,
    num_samples,
    figsize=(num_samples * 2, num_rows * 2),
    dpi=200
)

for i in range(num_samples):
    for k in range(num_rows):
        axes[k, i].imshow(decoded_all[i, k])
        axes[k, i].axis("off")

plt.tight_layout(pad=0.05)
save_path = "transition_grid.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
import os, sys
sys.path.append("ALAE")
import torch
import numpy as np
from tqdm import tqdm
from alae_ffhq_inference import load_model, encode, decode
from ffhq_utils import load_data, calculate_fid, decode_latents, get_activations
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError

np.random.seed(seed=1)
directory_ref = "conditional_flowmatching"
print("directory_ref: ", directory_ref)
model = load_model("ALAE/configs/ffhq.yaml", training_artifacts_dir="ALAE/training_artifacts/ffhq/")
model = model.to(device)
model.eval()

x0_path = os.path.join(directory_ref, "x0.npy")
xpred_path = os.path.join(directory_ref, "x_pred.npy")

x0 = np.load(x0_path)[:2000]
x_pred = np.load(xpred_path)[:2000]

print("x0: ", x0.shape)
print("x_pred: ", x_pred.shape)

num_samples = 10
num_steps = 5

# choose random identities
# inds = np.random.choice(x_pred.shape[1], num_samples, replace=False)
inds = (0, 17, 41, 12, 4, 14, 39, 44, 30, 34)
# inds = (0, 1, 13, 11, 4, 14, 16, 12, 8, 9)
# inds = (17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
# inds = (27, 28, 29, 30, 31, 32, 33, 34, 35, 36)
# inds = (37, 38, 39, 40, 41, 42, 43, 44, 45, 46)

# choose time indices
time_inds = np.linspace(0, x_pred.shape[0] - 1, num_steps).astype(int)

# select trajectories
selected_latents = x_pred[time_inds][:, inds]      # (steps, samples, 512)
selected_latents = selected_latents.transpose(1, 0, 2)  # (samples, steps, 512)

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
save_path = os.path.join(directory_ref, "transition_grid.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

fake_images = decode_latents(model, x_pred[-1], device)
INPUT_DATA = "ADULT" # MAN, WOMAN, ADULT, CHILDREN
TARGET_DATA = "CHILDREN" # MAN, WOMAN, ADULT, CHILDREN
train_size = 20000
test_size = 2000
_, Y_train, _, _ = load_data(train_size, test_size, INPUT_DATA, TARGET_DATA)
real_images = decode_latents(model, Y_train, device)

fid_score = calculate_fid(real_images, fake_images, device)
print("FID score:", fid_score)
import os, sys
sys.path.append("ALAE")
import torch
import numpy as np
from tqdm import tqdm
from alae_ffhq_inference import load_model, encode, decode
from ffhq_data_loading_utils import load_data
from matplotlib import pyplot as plt

DIM = 512
INPUT_DATA = "ADULT" # MAN, WOMAN, ADULT, CHILDREN
TARGET_DATA = "CHILDREN" # MAN, WOMAN, ADULT, CHILDREN

train_size = 60000
test_size = 10000

X_train, Y_train, X_test, Y_test = load_data(train_size, test_size, INPUT_DATA, TARGET_DATA)

print("X_train: ", X_train.shape)
print("Y_train: ", Y_train.shape)
print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)


model = load_model("ALAE/configs/ffhq.yaml", training_artifacts_dir="ALAE/training_artifacts/ffhq/")

# images
mapped = Y_test.clone()
number_of_samples = 3
decoded_all = []
with torch.no_grad():
    for k in range(number_of_samples):
        decoded_img = decode(model, mapped[k:k+1])
        decoded_img = ((decoded_img * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()
        decoded_all.append(decoded_img)
        
decoded_all = np.stack(decoded_all, axis=1)
print("decoded_all: ", decoded_all.shape)

imgs = decoded_all[0]
n = imgs.shape[0]

fig, axes = plt.subplots(1, n, figsize=(n*3, 3), dpi=200)

for k in range(n):
    axes[k].imshow(imgs[k])
    axes[k].axis("off")

plt.tight_layout()
plt.savefig("figures_decoded.png", dpi=300, bbox_inches="tight")
plt.close()
import os, sys
sys.path.append("ALAE")

import torch
import numpy as np

from tqdm import tqdm
from alae_ffhq_inference import load_model, encode, decode
from data_loading_utils import load_data

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


model = load_model("/ALAE/configs/ffhq.yaml", training_artifacts_dir="/ALAE/training_artifacts/ffhq/")

# images
mapped = Y_test.clone()

decoded_all = []
with torch.no_grad():
    for k in range(number_of_samples):
        decoded_img = decode(model, mapped[:, k])
        decoded_img = ((decoded_img * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()
        decoded_all.append(decoded_img)
        
decoded_all = np.stack(decoded_all, axis=1)
print("decoded_all: ", decoded_all.shape)

# plot
fig, axes = plt.subplots(10, number_of_samples+1, figsize=(number_of_samples+1, 10), dpi=200)

for i, ind in enumerate(range(10)):
    ax = axes[i]
    ax[0].imshow(inp_images[ind])
    for k in range(number_of_samples):
        ax[k+1].imshow(decoded_all[ind, k])
        
        ax[k+1].get_xaxis().set_visible(False)
        ax[k+1].set_yticks([])
        
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_yticks([])

fig.tight_layout(pad=0.05)
plt.savefig("figures_decoded", dpi=300, bbox_inches="tight")
plt.close(fig)
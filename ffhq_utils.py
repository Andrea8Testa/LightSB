import torch
import numpy as np
from alae_ffhq_inference import load_model, encode, decode
from tqdm import tqdm

def decode_latents(model, latents, device, batch_size=32):
    """
    latents: (N, 512)
    returns: numpy images (N, H, W, 3) uint8
    """
    model.eval()
    imgs = []
    print("decode_latents")
    with torch.no_grad():
        for i in tqdm(range(0, latents.shape[0], batch_size)):
            batch = latents[i:i+batch_size]

            z = torch.tensor(batch, dtype=torch.float32, device=device)
            img = decode(model, z)

            img = ((img * 0.5 + 0.5) * 255).clamp(0,255).to(torch.uint8)
            img = img.permute(0,2,3,1).cpu().numpy()

            imgs.append(img)

    return np.concatenate(imgs, axis=0)


from pytorch_fid.inception import InceptionV3
from scipy.linalg import sqrtm

def get_activations(images, model, device, batch_size=50):

    model.eval()
    activations = []

    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):

            batch = images[i:i+batch_size]

            batch = torch.tensor(batch).float() / 255.0
            batch = batch.permute(0,3,1,2).to(device)

            batch = torch.nn.functional.interpolate(
                batch, size=(299,299), mode="bilinear", align_corners=False
            )

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

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid

def load_data(train_size, test_size, input_data, target_data, seed=0):
    np.random.seed(seed=seed)
    latents = np.load("/home/tea1rng/unbalanced_workspace/baselines/LightSB/data/latents.npy")
    gender = np.load("/home/tea1rng/unbalanced_workspace/baselines/LightSB/data/gender.npy")
    age = np.load("/home/tea1rng/unbalanced_workspace/baselines/LightSB/data/age.npy")
    test_inp_images = np.load("/home/tea1rng/unbalanced_workspace/baselines/LightSB/data/test_images.npy")

    train_latents, test_latents = latents[:train_size], latents[-test_size:]
    train_gender, test_gender = gender[:train_size], gender[-test_size:]
    train_age, test_age = age[:train_size], age[-test_size:]

    if input_data == "MAN":
        x_inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        x_inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif input_data == "WOMAN":
        x_inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        x_inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif input_data == "ADULT":
        x_inds_train = np.arange(train_size)[
            (train_age >= 18).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        x_inds_test = np.arange(test_size)[
            (test_age >= 18).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    elif input_data == "CHILDREN":
        x_inds_train = np.arange(train_size)[
            (train_age < 18).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        x_inds_test = np.arange(test_size)[
            (test_age < 18).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    x_data_train = train_latents[x_inds_train]
    x_data_test = test_latents[x_inds_test]

    if target_data == "MAN":
        y_inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        y_inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif target_data == "WOMAN":
        y_inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        y_inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif target_data == "ADULT":
        y_inds_train = np.arange(train_size)[
            (train_age >= 18).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        y_inds_test = np.arange(test_size)[
            (test_age >= 18).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    elif target_data == "CHILDREN":
        y_inds_train = np.arange(train_size)[
            (train_age < 18).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        y_inds_test = np.arange(test_size)[
            (test_age < 18).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    y_data_train = train_latents[y_inds_train]
    y_data_test = test_latents[y_inds_test]

    X_train = torch.tensor(x_data_train)
    Y_train = torch.tensor(y_data_train)

    X_test = torch.tensor(x_data_test)
    Y_test = torch.tensor(y_data_test)

    return X_train, Y_train, X_test, Y_test
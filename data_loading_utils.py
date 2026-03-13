import torch
import numpy as np

def load_data(train_size, test_size, input_data, target_data, seed=0):
    np.random.seed(seed=seed)
    latents = np.load("data/latents.npy")
    gender = np.load("data/gender.npy")
    age = np.load("data/age.npy")
    test_inp_images = np.load("data/test_images.npy")


    train_latents, test_latents = latents[:train_size], latents[train_size:]
    train_gender, test_gender = gender[:train_size], gender[train_size:]
    train_age, test_age = age[:train_size], age[train_size:]

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
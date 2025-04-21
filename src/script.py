# imports
import json

import torch
import utils
from torch.utils.data import DataLoader


# declare filenames
filename_3x3 = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/310x310_model/dataset_3x3_10K.h5"
filename_6x6 = "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/310x310_model/dataset_6x6_10K.h5"

filenames_dict = {
    filename_3x3 :   8000,
    filename_6x6 :   8000
}

# get normalization constants
x_mean, x_std, y_mean, y_std = utils.get_normalization_constants(filenames_dict)
print("Debug")
# make training and test datasets
dataset_3x3_train = utils.HartreeFockDataset(filename_3x3, 
                                       indices=(0, 8000), 
                                       transform = lambda x : utils.normalize(x, mean = x_mean, std_dev = x_std), 
                                       target_transform = lambda y : utils.normalize(y, mean = y_mean, std_dev = y_std)
                                       )
print(f"\nTraining dataset length: {dataset_3x3_train.__len__()}\n")

dataset_6x6_train = utils.HartreeFockDataset(filename_6x6, 
                                       indices=(0, 8000), 
                                       transform = lambda x : utils.normalize(x, mean = x_mean, std_dev = x_std), 
                                       target_transform = lambda y : utils.normalize(y, mean = y_mean, std_dev = y_std)
                                       )
print(f"\nTraining dataset length: {dataset_6x6_train.__len__()}\n")

dataset_3x3_test = utils.HartreeFockDataset(filename_3x3, 
                                       indices=(8000, 10000), 
                                       transform = lambda x : utils.normalize(x, mean = x_mean, std_dev = x_std), 
                                       target_transform = lambda y : utils.normalize(y, mean = y_mean, std_dev = y_std)
                                       )
print(f"\nTest dataset length: {dataset_3x3_test.__len__()}\n")

dataset_6x6_test = utils.HartreeFockDataset(filename_6x6, 
                                       indices=(8000, 10000), 
                                       transform = lambda x : utils.normalize(x, mean = x_mean, std_dev = x_std), 
                                       target_transform = lambda y : utils.normalize(y, mean = y_mean, std_dev = y_std)
                                       )
print(f"\nTest dataset length: {dataset_6x6_test.__len__()}\n")


# training and test dataloaders
train_3x3_dataloader = DataLoader(
    dataset_3x3_train, batch_size=50, shuffle=True, pin_memory=True, num_workers=3
)
train_6x6_dataloader = DataLoader(
    dataset_6x6_train, batch_size=50, shuffle=True, pin_memory=True, num_workers=3
)
test_3x3_dataloader = DataLoader(
    dataset_3x3_test, batch_size=50, shuffle=True, pin_memory=True, num_workers=3
)
test_6x6_dataloader = DataLoader(
    dataset_6x6_test, batch_size=50, shuffle=True, pin_memory=True, num_workers=3
)

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# instantiate model
model = utils.HF_310x310_Model(filenames=[
    "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/positional_embeddings/true_momentum_3x3.pt",
    "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/positional_embeddings/true_momentum_6x6.pt",
    "/blue/yujiabin/awwab.azam/hartree-fock-code/datasets/positional_embeddings/true_momentum_9x9.pt"
], num_layers=3, input_dim=64, embed_dim=128, num_heads=8, dim_feedforward=512, device=device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}\n")

# Loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction = "sum")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_losses = []
test_losses = []
epochs = 1000

# train the model and record the loss for the training and test datasets
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_losses.append(
        utils.train_loop([train_3x3_dataloader, train_6x6_dataloader], model, loss_fn, optimizer, device)
    )
    test_losses.append(utils.test_loop([test_3x3_dataloader, test_6x6_dataloader], model, loss_fn, device))
print("Done!\n")

print(f"Training Losses (1-{epochs}): {train_losses}")
print(f"Test Losses (1-{epochs}): {test_losses}\n")

utils.plot_losses(train_losses, test_losses)

# saving model
torch.save(
    model.state_dict(),
    "Band_Projected_In_PlaneWaves_model.pth",
)

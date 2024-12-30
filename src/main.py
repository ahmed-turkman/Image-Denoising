import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parameters
batch_size = 64
num_epochs = 50
learning_rate = 0.001
train_portion = 0.7 # Train dataset will be 70% of the whole dataset
val_portion = 0.15
test_portion = 0.15


# Dataset and DataLoader
img_dir = "./dataset/lfwcrop_color/faces"
dataset = NoiseImagesDataset(img_dir)
indices = list(range(len(dataset)))
np.random.shuffle(indices)
split1 = int(len(dataset) * train_portion)
split2 = int(len(dataset) * (train_portion + val_portion))
train_indices = indices[:split1]
val_indices = indices[split1:split2]
test_indices = indices[split2:]

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))


# Model, Loss, Optimizer
model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Starting training..
train_losses, val_losses = train(num_epochs=50, train_loader=train_loader, val_loader=val_loader, device=device)

# Plot Train MSE Loss vs Validation MSE Loss
plot_train_vs_val_loss(train_losses, val_losses)

# Compute and print the MSE loss on the test set
compute_test_loss(model, test_loader, device)

# Visualize the model's performance on three random imagest from the test set
visualize_random_samples(model, test_loader, device)
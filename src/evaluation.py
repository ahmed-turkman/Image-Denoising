import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_train_vs_val_loss(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs iteration')
    plt.legend()


def compute_test_loss(model, test_loader, device):
    with torch.no_grad():
        total = 0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total += loss.item()

        test_loss = total / len(test_loader)

    print(f'Average Loss on the test images: {test_loss:.4f} ')


def visualize_random_samples(model, test_loader, device):
    with torch.no_grad():
        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            true = np.array(labels[0].to('cpu'))
            noisy = np.array(inputs[0].to('cpu'))
            denoised = outputs[0].to('cpu')
            denoised = denoised.detach().numpy()
            noisy = noisy.transpose(1, 2, 0)
            true = true.transpose(1, 2, 0)
            denoised = denoised.transpose(1, 2, 0)

            axes[i, 0].imshow(true, cmap='gray')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(noisy, cmap='gray')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(denoised, cmap='gray')
            axes[i, 2].axis('off')

            if i == 0:
                axes[i, 0].set_title('True Images', fontsize=12, weight='bold')
                axes[i, 1].set_title('Noisy Images', fontsize=12, weight='bold')
                axes[i, 2].set_title('Denoised Images', fontsize=12, weight='bold')

            if i == 2:
                break

    plt.suptitle("Comparison of True, Noisy, and Denoised Images", fontsize=14)
    plt.tight_layout()
    plt.show()
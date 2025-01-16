import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 3x32x32 -> 64x16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x16x16 -> 128x8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128x8x8 -> 256x4x4
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)  # Log-variance

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 256x4x4 -> 128x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x8x8 -> 64x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 64x16x16 -> 3x32x32
            nn.Sigmoid()  # Output in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) using N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        dec_input = self.fc_dec(z).view(-1, 256, 4, 4)
        recon_x = self.decoder(dec_input)

        return recon_x, mu, logvar

def loss(x, recon_x, mu, logvar):
    """VAE Loss = Reconstruction Loss + KL Divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

class Cifar10(Dataset):

    def __init__(self, root, transforms=None):

        self.image_list = []
        for image_file in os.listdir(root):
            if image_file.endswith('.png'):
                self.image_list.append(os.path.join(root, image_file))

        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        if self.transforms:
            image = self.transforms(image)
        return image


def visualize_reconstruction(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            recon_x, _, _ = model(x)
            break  # Take one batch

    # Denormalize for visualization
    x = x * 0.5 + 0.5  # Assuming normalization with mean=0.5, std=0.5
    recon_x = recon_x * 0.5 + 0.5

    # Convert to CPU and numpy
    x = x.cpu().numpy()
    recon_x = recon_x.cpu().numpy()

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(x[i].transpose(1, 2, 0))  # Original
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_x[i].transpose(1, 2, 0))  # Reconstructed
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=14)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=14)
    plt.show()

def train_vae_on_cifar10(epochs=10, batch_size=128, latent_dim=128):
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = Cifar10(root=r'D:\pyproject\Understanding-Diffusion-Models-A-Unified-Perspective\dataset\cifar_train',
                            transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, and device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

        for x in progress_bar:
            x = x.to(device)

            # Forward pass
            recon_x, mu, logvar = model(x)

            # Compute loss
            loss_value = loss(x, recon_x, mu, logvar)
            epoch_loss += loss_value.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=loss_value.item())

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

        # Visualize reconstruction at the end of each epoch
        visualize_reconstruction(model, train_loader, device)

if __name__ == '__main__':
    train_vae_on_cifar10(epochs=10, batch_size=128, latent_dim=128)

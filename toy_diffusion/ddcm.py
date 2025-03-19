import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import load_data
from unet import UNet


device = 'cuda'
T = 10
alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
beta = np.sqrt(1 - alpha**2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha**2)
sigma = beta.copy()
K = 8192
codebook = torch.randn(size=(K, 1, 32, 32), device=device)

alpha = torch.from_numpy(alpha).to(device)
beta = torch.from_numpy(beta).to(device)
bar_alpha = torch.from_numpy(bar_alpha).to(device)
bar_beta = torch.from_numpy(bar_beta).to(device)
sigma = torch.from_numpy(sigma).to(device)

class DiffusionModel:

    def __init__(self, T: int, model: nn.Module, device: str):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

    def training(self, batch, optimizer):
        """
        Algorithm 1 in Denoising Diffusion Probabilistic Models
        """

        x_samples = batch.to(self.device)
        batch_size = x_samples.size(0)

        z_samples = codebook[np.random.choice(K, batch_size)].to(self.device)

        for t in range(self.T):
            t = T - t - 1

            bt = torch.tensor([t], device=self.device)
            mp = self.function_approximator(z_samples, bt)

            x0 = (z_samples - bar_beta[t] * mp) / bar_alpha[t]
            sims = torch.einsum('kuwv,buwv->kb', codebook, x_samples - x0)
            idxs = sims.argmax(0)
            z_samples -= beta[t] ** 2 / bar_beta[t] * mp
            z_samples /= alpha[t]
            z_samples += codebook[idxs] * sigma[t]
        loss = nn.MSELoss()(z_samples, x_samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32),
                 use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """

        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]),
                        device=self.device)
        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t

            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(
                1 - alpha_bar_t)) * self.function_approximator(x, t - 1))
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
        return x


if __name__ == "__main__":
    device = 'cuda'
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    diffusion_model = DiffusionModel(10, model, device)
    trainX, testX = load_data()

    epochs = 1
    # Training
    for epoch in range(epochs):
        pbar = tqdm(trainX, desc=f"{epoch + 1}/{epochs}", unit="batch")
        for batch, _ in pbar:
            loss = diffusion_model.training(batch, optimizer)
            pbar.set_postfix(loss=f'{loss:.4f}')

    # Plot results
    nb_images = 81
    samples = diffusion_model.sampling(n_samples=nb_images, use_tqdm=False)
    plt.figure(figsize=(17, 17))
    for i in range(nb_images):
        plt.subplot(9, 9, 1 + i)
        plt.axis('off')
        plt.imshow(samples[i].squeeze(0).clip(0, 1).data.cpu().numpy(),
                   cmap='gray')
    plt.savefig(f'./samples.png')
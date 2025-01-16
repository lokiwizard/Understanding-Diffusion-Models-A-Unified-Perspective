import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, encoder, decoder, encoder_dim=512, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = encoder  # 自定义的编码器
        self.decoder = decoder  # 自定义的解码器
        self.mu = nn.Linear(encoder_dim, latent_dim)  # 均值
        self.logvar = nn.Linear(encoder_dim, latent_dim)  # 对数方差

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)     # 采样 epsilon
        return mu + eps * std           # 重参数化技巧

    def forward(self, x):
        mu = self.mu(self.encoder(x))    # 编码器输出z的均值和对数方差
        logvar = self.logvar(self.encoder(x))

        z = self.reparameterize(mu, logvar)  # 采样潜在向量
        recon_x = self.decoder(z)       # 解码重建输入
        return recon_x, mu, logvar
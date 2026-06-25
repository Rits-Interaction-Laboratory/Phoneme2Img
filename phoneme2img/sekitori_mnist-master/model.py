import torch
import torch.nn as nn


class VAE_fc(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE_fc, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(200, z_dim)
        self.fc_logvar = nn.Linear(200, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 784))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar, num_samples=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), device=mu.device)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 784)


class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 400 ), #400ch 
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, z_dim)
        self.fc_logvar = nn.Linear(400, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 1, 28, 28))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar, num_samples=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), device=mu.device)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 784)

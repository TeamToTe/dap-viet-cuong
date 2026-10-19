"""
Fence GAN Model Plugin.
Minimax boundary loss forcing Generator to place samples on the boundary fence of normal distribution.
Based on Ngo et al., IJCNN 2019: 'Fence GAN: Towards Better Anomaly Detection'.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..core.registry import register_model
from ..core.base_model import BaseGenerativeModel


class FenceGANGen(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FenceGANDisc(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LeakyReLU(0.2), nn.Dropout(0.2)])
            prev = h
        layers.extend([nn.Linear(prev, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("fence_gan")
class FenceGAN(BaseGenerativeModel):
    """
    Fence GAN Model Plugin.
    Enforces boundary generation with gamma target, boundary constraint, and dispersion loss.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        gamma: float = 0.5,
        alpha: float = 0.1,
        beta: float = 1.0,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(input_dim=input_dim, device=device, **kwargs)
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.G = FenceGANGen(latent_dim, [64, 64], input_dim).to(self.device)
        self.D = FenceGANDisc(input_dim, [64, 32]).to(self.device)

    def dispersion_loss(self, fake: torch.Tensor) -> torch.Tensor:
        B = fake.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=self.device)
        diff = fake.unsqueeze(1) - fake.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        sigma_sq = 2.0 * fake.size(1)
        k = torch.exp(-dist_sq / sigma_sq)
        mask = 1.0 - torch.eye(B, device=self.device)
        return torch.sum(k * mask) / (B * (B - 1) + 1e-8)

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 2e-4,
        **kwargs
    ) -> Dict[str, Any]:
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=(len(X_train) > batch_size)
        )
        opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        bce = nn.BCELoss()

        history = []
        for epoch in range(1, epochs + 1):
            loss_d_cum, loss_g_cum = 0.0, 0.0
            for (real_x,) in loader:
                real_x = real_x.to(self.device)
                B = real_x.size(0)

                # Train Discriminator
                opt_D.zero_grad()
                d_real = self.D(real_x)
                l_real = bce(d_real, torch.ones(B, 1, device=self.device) * 0.9)
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_x = self.G(z)
                d_fake = self.D(fake_x.detach())
                l_fake = bce(d_fake, torch.zeros(B, 1, device=self.device))
                loss_D = l_real + l_fake
                loss_D.backward()
                opt_D.step()

                # Train Generator (Fence Objective)
                opt_G.zero_grad()
                d_g = self.D(fake_x)
                loss_target = torch.mean((d_g - self.gamma) ** 2)
                loss_bound = torch.mean(torch.abs(d_g - self.gamma))
                loss_disp = self.dispersion_loss(fake_x)
                loss_G = loss_target + self.beta * loss_bound + self.alpha * loss_disp
                loss_G.backward()
                opt_G.step()

                loss_d_cum += loss_D.item()
                loss_g_cum += loss_G.item()

            avg_d = loss_d_cum / max(1, len(loader))
            avg_g = loss_g_cum / max(1, len(loader))
            history.append({"loss_D": avg_d, "loss_G": avg_g})

            interval = 20 if epochs >= 50 else (10 if epochs >= 25 else 5)
            if epoch % interval == 0 or epoch == epochs:
                print(f"  [FenceGAN] Epoch {epoch:03d}/{epochs} - Loss D: {avg_d:.4f} | Loss G (Boundary Fence): {avg_g:.4f}")

        return {
            "loss_D": [h["loss_D"] for h in history],
            "loss_G": [h["loss_G"] for h in history],
            "history": history
        }

    def sample(self, n_samples: int, **kwargs) -> np.ndarray:
        """Sample synthetic boundary anomalies from the Generator."""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            return self.G(z).cpu().numpy()

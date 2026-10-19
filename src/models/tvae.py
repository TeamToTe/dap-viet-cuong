"""
Tabular Variational Autoencoder (TVAE) Plugin.
Generates anomalies via Latent Outlier Tail Sampling and Directional Latent Shift.
Based on Xu et al., NeurIPS 2019 / Kingma & Welling, 2013.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..core.registry import register_model
from ..core.base_model import BaseGenerativeModel


class TVAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)])
            prev = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class TVAEDecoder(nn.Module):
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


@register_model("tvae")
class TVAE(BaseGenerativeModel, nn.Module):
    """
    TVAE Model Plugin.
    Learns normal tabular sensor distribution manifold.
    Samples synthetic anomalies via latent sphere tail perturbation + directional shift.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        beta: float = 0.05,
        device: Optional[str] = None,
        **kwargs
    ):
        BaseGenerativeModel.__init__(self, input_dim=input_dim, device=device, **kwargs)
        nn.Module.__init__(self)

        self.latent_dim = min(latent_dim, max(4, input_dim))
        self.beta = beta
        encoder_hidden = hidden_dims or [64, 32]
        decoder_hidden = list(reversed(encoder_hidden))

        self.encoder = TVAEEncoder(input_dim, encoder_hidden, self.latent_dim)
        self.decoder = TVAEDecoder(self.latent_dim, decoder_hidden, input_dim)
        self.to(self.device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_func(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        mse = nn.functional.mse_loss(recon, x)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return mse + self.beta * kl

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        **kwargs
    ) -> Dict[str, Any]:
        self.train()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=(len(X_train) > batch_size)
        )
        opt = optim.Adam(self.parameters(), lr=lr)

        loss_history = []
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (bx,) in loader:
                bx = bx.to(self.device)
                opt.zero_grad()
                recon, mu, logvar = self(bx)
                loss = self.loss_func(bx, recon, mu, logvar)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(loader))
            loss_history.append(avg_loss)
            interval = 20 if epochs >= 50 else (10 if epochs >= 25 else 5)
            if epoch % interval == 0 or epoch == epochs:
                print(f"  [TVAE] Epoch {epoch:03d}/{epochs} - Reconstruction & KL Loss: {avg_loss:.4f}")

        return {"final_loss": loss_history[-1], "loss_history": loss_history, "loss": loss_history}

    def sample_tail_anomalies(self, n: int, min_sigma: float = 2.5, max_sigma: float = 4.2) -> np.ndarray:
        """Sample from the tails of the Gaussian latent manifold."""
        self.eval()
        with torch.no_grad():
            u = torch.randn(n, self.latent_dim, device=self.device)
            u = u / torch.norm(u, dim=1, keepdim=True)
            r = torch.empty(n, 1, device=self.device).uniform_(min_sigma, max_sigma)
            return self.decoder(u * r).cpu().numpy()

    def sample_directional_anomalies(self, X_ref: np.ndarray, n: int, drift: float = 2.8) -> np.ndarray:
        """Encode normal instances and shift their latent positions along defect directions."""
        self.eval()
        with torch.no_grad():
            ref = torch.tensor(X_ref[:min(n, len(X_ref))], dtype=torch.float32, device=self.device)
            mu, _ = self.encoder(ref)
            if len(mu) < n:
                reps = (n // len(mu)) + 1
                mu = mu.repeat(reps, 1)[:n]
            drift_vec = torch.zeros_like(mu)
            drift_vec[:, 0] = drift
            if self.latent_dim > 1:
                drift_vec[:, 1] = drift * 0.6
            return self.decoder(mu + drift_vec + torch.randn_like(mu) * 0.25).cpu().numpy()

    def sample(self, n_samples: int, **kwargs) -> np.ndarray:
        """Balanced synthesis of tail anomalies and directional latent perturbations."""
        half = n_samples // 2
        tail_samples = self.sample_tail_anomalies(half)
        # Generate remaining half via tail sampling if reference not provided
        X_ref = kwargs.get("X_ref", None)
        if X_ref is not None and len(X_ref) > 0:
            dir_samples = self.sample_directional_anomalies(X_ref, n_samples - half)
        else:
            dir_samples = self.sample_tail_anomalies(n_samples - half, min_sigma=3.0, max_sigma=5.0)
        return np.vstack([tail_samples, dir_samples])

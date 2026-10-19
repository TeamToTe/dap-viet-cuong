"""
TabDDPM (Tabular Denoising Diffusion Probabilistic Model) Plugin.
Based on Kotelnikov et al., NeurIPS 2022.
Generates tabular anomalies via guided/perturbed reverse diffusion steps.
"""

import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..core.registry import register_model
from ..core.base_model import BaseGenerativeModel


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TabDDPMBackbone(nn.Module):
    def __init__(self, input_dim: int = 8, time_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_t1 = nn.Linear(time_dim, hidden_dim)
        self.b1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        self.b2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_rep = self.time_emb(t)
        h = self.fc_in(x) + self.fc_t1(t_rep)
        h = h + self.b1(h)
        h = h + self.b2(h)
        return self.fc_out(h)


@register_model("tab_ddpm")
class TabDDPM(BaseGenerativeModel):
    """
    TabDDPM Model Plugin.
    State-of-the-Art tabular generative model preserving non-linear inter-sensor correlation.
    Samples synthetic anomalies using late-step reverse perturbation guidance.
    """
    def __init__(
        self,
        input_dim: int,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        time_dim: int = 32,
        hidden_dim: int = 128,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(input_dim=input_dim, device=device, **kwargs)
        self.timesteps = timesteps
        self.model = TabDDPMBackbone(
            input_dim=input_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_var = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        **kwargs
    ) -> Dict[str, Any]:
        self.model.train()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=(len(X_train) > batch_size)
        )
        opt = optim.Adam(self.model.parameters(), lr=lr)
        mse = nn.MSELoss()

        history = []
        for epoch in range(1, epochs + 1):
            total_l = 0.0
            for (x_0,) in loader:
                x_0 = x_0.to(self.device)
                B = x_0.size(0)
                t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
                noise = torch.randn_like(x_0)
                x_t = self.sqrt_alphas_cumprod[t].view(-1, 1) * x_0 + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise
                pred_noise = self.model(x_t, t)
                loss = mse(pred_noise, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_l += loss.item()

            avg_loss = total_l / max(1, len(loader))
            history.append(avg_loss)
            interval = 20 if epochs >= 50 else (10 if epochs >= 25 else 5)
            if epoch % interval == 0 or epoch == epochs:
                print(f"  [TabDDPM] Epoch {epoch:03d}/{epochs} - Diffusion Score Loss: {avg_loss:.4f}")

        return {"final_loss": history[-1], "history": history, "loss": history}

    def sample(
        self,
        n_samples: int,
        perturb_timesteps: int = 15,
        drift_strength: float = 2.6,
        **kwargs
    ) -> np.ndarray:
        """Sample synthetic anomalies using reverse diffusion with late perturbation guidance."""
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, self.input_dim, device=self.device)
            for t_idx in reversed(range(self.timesteps)):
                t = torch.full((n_samples,), t_idx, device=self.device, dtype=torch.long)
                pred_noise = self.model(x, t)
                beta = self.betas[t_idx]
                alpha = self.alphas[t_idx]
                alpha_bar = self.alphas_cumprod[t_idx]

                mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise)

                # Inject guidance drift at late reverse steps for boundary anomaly steering
                if t_idx <= perturb_timesteps:
                    drift = torch.zeros_like(mean)
                    drift[:, 0] = drift_strength * (1.0 + 0.2 * torch.randn(n_samples, device=self.device))
                    if self.input_dim > 3:
                        drift[:, 3] = -drift_strength * 0.8 * (1.0 + 0.2 * torch.randn(n_samples, device=self.device))
                    mean = mean + (beta / torch.sqrt(1.0 - alpha_bar)) * drift

                if t_idx > 0:
                    x = mean + torch.sqrt(self.posterior_var[t_idx]) * torch.randn_like(x)
                else:
                    x = mean

        return x.cpu().numpy()

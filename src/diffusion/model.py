"""
Cold Diffusion MLP for embedding restoration.

Degradation schedule:
    x_t = (1 - λ_t) * E_full  +  λ_t * E_under    λ_t = t / T
    t=0   → x_0   = E_full    (clean)
    t=T-1 → x_{T-1} ≈ E_under (fully degraded)

Model predicts Δ = E_under − E_full given (x_t, E_under, t).
Restoration: E_restored = E_under − Δ_pred
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep t ∈ [0, T), with MLP projection."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class SwiGLUBlock(nn.Module):
    """
    Post-norm SwiGLU:
        fc1(in_dim → hidden*2) → split → SiLU(gate) * val
        → fc2(hidden → out_dim) + residual(in_dim → out_dim)
        → LayerNorm(out_dim)
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden * 2)
        self.fc2  = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.res  = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.fc1(x).chunk(2, dim=-1)
        h = F.silu(gate) * val
        return self.norm(self.fc2(h) + self.res(x))


class ColdDiffusionMLP(nn.Module):
    """
    Predicts the embedding deficit Δ = E_under − E_full.

    Args:
        emb_dim : embedding dimension D (depends on pooling strategy)
        hidden  : hidden width of SwiGLU blocks
        depth   : number of SwiGLU blocks
        T       : number of diffusion timesteps
        time_dim: dimension of time embedding
    """

    def __init__(
        self,
        emb_dim:  int = 512,
        hidden:   int = 1024,
        depth:    int = 4,
        T:        int = 100,
        time_dim: int = 256,
    ):
        super().__init__()
        self.T       = T
        self.emb_dim = emb_dim
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        in_dim = 2 * emb_dim + time_dim
        layers = [SwiGLUBlock(in_dim, hidden, hidden)]
        for _ in range(depth - 1):
            layers.append(SwiGLUBlock(hidden, hidden, hidden))
        self.blocks = nn.Sequential(*layers)

        self.out = nn.Linear(hidden, emb_dim)

    def _lambda(self, t: torch.Tensor) -> torch.Tensor:
        return (t.float() / self.T).view(-1, 1)

    def degrade(self, e_full, e_under, t):
        lam = self._lambda(t)
        return (1.0 - lam) * e_full + lam * e_under

    def forward(self, x_t, t, cond):
        t_emb = self.time_emb(t)
        inp   = torch.cat([x_t, cond, t_emb], dim=-1)
        h     = self.blocks(inp)
        return self.out(h)

    @torch.no_grad()
    def sample(self, e_under, steps=1):
        self.eval()
        device = e_under.device
        B = e_under.size(0)

        if steps == 1:
            t = torch.full((B,), self.T - 1, device=device, dtype=torch.long)
            delta = self(e_under, t, e_under)
            return e_under - delta

        # Multi-step cold sampling.
        # 시간표를 T-1 → 0 으로 (steps+1)개 상태로 쪼개고, 각 스텝에서
        #   (i) 현재 상태 x_{t_i}에서 E_f를 추정하고
        #   (ii) 다음 시간 t_{i+1}의 λ로 재열화한다.
        # 모델 호출 t와 재열화 λ가 같은 시간표의 연속 쌍을 쓰므로 어긋나지 않는다.
        # λ는 학습 규약(_lambda = t/T)과 동일하게 계산해 train/inference 분포를 맞춘다.
        x = e_under.clone()
        ts = torch.linspace(self.T - 1, 0, steps + 1, device=device).long()
        for i in range(steps):
            t_cur = ts[i]
            t = torch.full((B,), int(t_cur.item()), device=device, dtype=torch.long)
            delta = self(x, t, e_under)
            e_full_hat = e_under - delta
            t_next = int(ts[i + 1].item())
            if t_next > 0:
                lam_next = t_next / self.T          # 학습과 동일한 λ 규약
                x = (1.0 - lam_next) * e_full_hat + lam_next * e_under
            else:
                x = e_full_hat
        return x


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self):
        self.model.load_state_dict(self.shadow)

    def restore(self, original_sd: dict):
        self.model.load_state_dict(original_sd)

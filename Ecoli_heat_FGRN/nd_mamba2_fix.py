"""
mamba2-minimal (Fixed & Stabilized)
===================================

A minimal, single-file implementation of the Mamba-2 model in PyTorch.
Optimized for numerical stability and proper initialization.

Based on:
> Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

from dataclasses import dataclass
from typing import NamedTuple
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

Device = torch.device


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2(nn.Module):
    def __init__(self, d_model: int,
                 n_layer: int = 24,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 chunk_size: int = 64,
                 vocab_size: int = 50277,
                 pad_vocab_size_multiple: int = 16, ):
        super().__init__()
        args = Mamba2Config(d_model, n_layer, d_state, d_conv, expand,
                            headdim, chunk_size, vocab_size, pad_vocab_size_multiple)
        self.args = args

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
        )

        # Fix: Proper initialization instead of torch.empty
        self.dt_bias = nn.Parameter(torch.rand(args.nheads))
        self.A_log = nn.Parameter(torch.randn(args.nheads))
        self.D = nn.Parameter(torch.ones(args.nheads))

        self.norm = RMSNorm(args.d_inner)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False)

        # Apply specific weight initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize A_log to ensure A starts in a reasonable range (e.g. -1 to -10)
        # Using a logarithmic distribution for timescales is standard in SSMs
        nn.init.normal_(self.A_log, mean=0, std=0.1)

        # Initialize dt_bias to be roughly uniform
        nn.init.uniform_(self.dt_bias, -2.0, 2.0)

        # D is a skip connection, initialize to 1
        nn.init.ones_(self.D)

        # Linear layers usually benefit from Kaiming or Xavier, but default is often fine.
        # Here we ensure conv1d weights are not garbage
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

    def forward(self, u: Tensor, h=None):
        if h:
            return self.step(u, h)

        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        xBC = F.silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state,
                  self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)

        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=x.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: Tensor, h: InferenceCache):
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = F.silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state,
                  self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2"""
    assert x.shape[1] % chunk_size == 0

    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp",
                         C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization
        Optimized for numerical stability (always computes in float32).
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        # Ensure computation is done in float32 to prevent NaN in mixed precision training
        dtype = x.dtype
        x = x.float()

        if z is not None:
            z = z.float()
            x = x * F.silu(z)

        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x * rrms * self.weight.float()

        return out.to(dtype)


# Wrapper classes
class BaseNdMamba2(nn.Module):
    def __init__(self, cin,  cout, mamba_dim, **mamba2_args):
        super().__init__()
        assert mamba_dim % 64 == 0, "mamba_dim must be divisible by 64"
        self.fc_in = nn.Linear(cin, mamba_dim, bias=False)
        self.mamba2_for = Mamba2(mamba_dim, **mamba2_args)
        self.mamba2_back = Mamba2(mamba_dim, **mamba2_args)
        self.fc_out = nn.Linear(mamba_dim, cout, bias=False)

        # Initialize linear layers
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

class NdMamba2_1d(BaseNdMamba2):
    def __init__(self, cin, cout, cmid,  **mamba2_args):
        super().__init__(cin, cout, cmid, **mamba2_args)

    def forward(self, x):
        l = x.shape[2]
        pad_len = (64 - x.shape[2] % 64) % 64
        x = F.pad(x, (0, pad_len))
        x = rearrange(x, 'b c l-> b l c')
        x = self.fc_in(x)
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)
        x = rearrange(x, 'b l c -> b c l')
        x = x[:, :, :l]
        return x

from typing import override
from torch import Tensor, nn
import torch
import numpy as np


# Network hyperparam
dim = 65
dimT = 50
L = 4
mode = 8
width = 32
lr = 5e-4
du = 2  # 2 for x, y components.
da = du + 2  # 2 for viscosity and pressure

# PDE param
T = (5, 10)
re = 500
alpha = 1
beta = 1
k = 1

# NN


# Fourier Convolution Operator
class FCO(nn.Module):
    def __init__(self, mode: int) -> None:
        super().__init__()
        self.mode: int = mode
        self.R: nn.Module = nn.Linear(self.mode, self.mode, dtype=torch.complex64)

    @override
    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        fft = torch.fft.fftn(x, dim=())
        x = self.R(fft)
        x = torch.fft.ifftn(x, s=shape)
        return x


class PINOLayer(nn.Module):
    def __init__(self, mode: int, width: int) -> None:
        super().__init__()

        self.k: nn.Module = FCO(mode)
        self.w: nn.Module = nn.Linear(width, width)
        self.sigma: nn.Module = nn.GELU()

    @override
    def forward(self, x: Tensor) -> Tensor:
        w = self.w(x)
        k = self.k(x)
        x = self.sigma(w + k)
        return x


class PINO(nn.Module):
    def __init__(self, da: int, du: int, mode: int, width: int, L: int) -> None:
        super().__init__()

        # uplift
        R = nn.Linear(da, width)
        # projection
        Q = nn.Linear(width, du)

        layers = [PINOLayer(mode, width) for _ in range(L)]
        self.core: nn.Module = nn.Sequential(
            R,
            *layers,
            Q,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.core(x)


pino = PINO(da, du, mode, width, L)

Batch = 2
test_input = torch.rand(Batch, dim, dim, da)

output = pino(test_input)

loss = output.sum()
loss.backward()
# Loss


# Boundary Conditions

INT = np.s_[1:-1, 1:-1]
XNT = np.s_[2:  , 1:-1]
XMT = np.s_[0:-2, 1:-1]
YNT = np.s_[1:-1, 2:  ]
YMT = np.s_[1:-1, 0:-2]

# No slip + top shift
def ldc_bc(u, K):
    u[1:-1, 0, 0] = K
    u[1:-1, 0, 1] = 0

    u[1:-1, -1, :] = 0
    u[ 0, 1:-1, :] = 0
    u[-1, 1:-1, :] = 0

def neumann_bc(p):
    p[1:-1,  0] = p[1:-1,  1]
    p[1:-1, -1] = p[1:-1, -2]
    p[ 0, 1:-1] = p[ 1, 1:-1]
    p[-1, 1:-1] = p[-2, 1:-1]

import torch
import torch.nn as nn

# chunk operation frequently used in VAE or Affine coupling
# split feature dim if x has shapes [B, d] or [B, N, d]
# split channel dim if x has shapes [B, C, H, W] or [B, N, C, H, W]
def chunk_two(x, split_dim=None):
    split_dim = (-3 if x.dim() > 3 else -1) \
            if split_dim is None else split_dim
    return x.chunk(2, split_dim)

def cat_two(x1, x2, split_dim=None):
    split_dim = (-3 if x1.dim() > 3 else -1) \
            if split_dim is None else split_dim
    return torch.cat([x1, x2], split_dim)

# sum except batch dim
# sum feature dim if x has shapes [B, d] or [B, N, d]
# sum channel dim if x has shapes [B, C, H, W] or [B, N, C, H, W]
def sum_except_batch(x, dims=None):
    dims = ([-3, -2, -1] if x.dim() > 3 else [-1]) \
            if dims is None else dims
    return x.sum(dims)

def to_numpy(x):
    return x.cpu().data.numpy()

def meshgrid_around(X, nx, ny, margin=0.1):
    x, y = torch.meshgrid(
            torch.linspace(
                X[:,0].min().item()-margin,
                X[:,0].max().item()+margin,
                nx).to(X.device),
            torch.linspace(
                X[:,1].min().item()-margin,
                X[:,1].max().item()+margin,
                ny).to(X.device))
    return torch.cat([x.reshape(nx*ny, 1), y.reshape(nx*ny, 1)], -1), x, y

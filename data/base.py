import torch
from torch.distributions import Dirichlet, Categorical
import math

def sample_partitions(B, N, K, alpha=1.0, rand_K=True, device='cpu'):
    pi = Dirichlet(alpha*torch.ones(K)).sample([B]).to(device)
    if rand_K:
        to_use = (torch.rand(B, K) < 0.5).float().to(device)
        to_use[...,0] = 1
        pi = pi * to_use
        pi = pi/pi.sum(1, keepdim=True)
    labels = Categorical(probs=pi).sample([N]).to(device)
    labels = labels.transpose(0,1).contiguous()
    return labels

def sample_anchors(B, N, mask=None, device='cpu'):
    if mask is None:
        return torch.randint(N, [B]).to(device)
    else:
        anc_idxs = torch.zeros(B, dtype=torch.int64).to(device)
        for b in range(B):
            if mask[b].sum() < N:
                idx_pool = mask[b].bitwise_not().nonzero().view(-1)
                anc_idxs[b] = idx_pool[torch.randint(len(idx_pool), [1])]
        return anc_idxs

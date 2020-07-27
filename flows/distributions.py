import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.tensor import chunk_two, sum_except_batch as sumeb

"""
copied & adapted from
https://github.com/bayesiains/nsf
"""

class Distribution(nn.Module):
    def __init__(self, shape,
            use_context=False, context_enc=nn.Identity()):
        super().__init__()
        self.shape = [shape] if isinstance(shape, int) \
                else list(shape)
        self.use_context = use_context
        self.context_enc = context_enc

    # infer shape for sampling
    def infer_shape(self, num_samples=None):
        if num_samples is None:
            return [1] + self.shape
        elif isinstance(num_samples, int):
            return [num_samples] + self.shape
        else:
            return list(num_samples) + self.shape

    def sample(self, num_samples=None, context=None):
        raise NotImplementedError

    def mean(self, context=None):
        raise NotImplementedError

    def log_prob(self, x, context=None):
        raise NotImplementedError

class Normal(Distribution):
    def unit_log_prob(self, x):
        return -0.5*x**2 - 0.5*math.log(2*math.pi)

    def sample(self, num_samples=None, context=None, device='cpu'):
        if self.use_context and context is not None:
            mu, sigma = chunk_two(self.context_enc(context))
            sigma = F.softplus(sigma) + 1e-5
            eps = torch.randn_like(mu)
            x = mu + sigma * eps
            lp = sumeb(self.unit_log_prob(eps) - sigma.log())
            return x, lp
        else:
            eps = torch.randn(self.infer_shape(num_samples), device=device)
            lp = sumeb(self.unit_log_prob(eps))
            return eps, lp

    def mean(self, context=None, device='cpu'):
        if self.use_context and context is not None:
            mu, sigma = chunk_two(self.context_enc(context))
            sigma = F.softplus(sigma) + 1e-5
            lp = sumeb(self.unit_log_prob(torch.zeros_like(mu)) \
                    - sigma.log())
            return mu, lp
        else:
            mu = torch.zeros(self.infer_shape(), device=device)
            lp = sumeb(self.unit_log_prob(mu))
            return mu, lp

    def log_prob(self, x, context=None):
        if self.use_context and context is not None:
            mu, sigma = chunk_two(self.context_enc(context))
            sigma = F.softplus(sigma) + 1e-5
            eps = (x - mu) / sigma
            return sumeb(self.unit_log_prob(eps) - sigma.log())
        else:
            return sumeb(self.unit_log_prob(x))

class Bernoulli(Distribution):
    def sample(self, num_samples=None, context=None, device='cpu'):
        if self.use_context and context is not None:
            logits = self.context_enc(context)
        else:
            logits = torch.zeros(self.infer_shape(num_samples), device=device)
        x = torch.bernoulli(torch.sigmoid(logits))
        lp = -sumeb(F.binary_cross_entropy_with_logits(
            logits, x, reduction='none'))
        return x, lp

    def mean(self, context=None, device='cpu'):
        if self.use_context and context is not None:
            logits = self.context_enc(context)
        else:
            logits = torch.zeros(self.infer_shape(), device=device)
        x = torch.sigmoid(logits)
        lp = -sumeb(F.binary_cross_entropy_with_logits(
            logits, x, reduction='none'))
        return x, lp

    def log_prob(self, x, context=None):
        if self.use_context and context is not None:
            logits = self.context_enc(context)
        else:
            logits = torch.zeros_like(x)
        return -sumeb(F.binary_cross_entropy_with_logits(
            logits, x, reduction='none'))

class FlowDistribution(Distribution):
    def __init__(self, transform, base):
        super().__init__(base.shape)
        self.transform = transform
        self.base = base

    def sample(self, num_samples=None, context=None, device='cpu'):
        x, lp = self.base.sample(num_samples, context, device)
        x, logdet = self.transform.inverse(x, context)
        return x, lp - logdet

    def mean(self, context=None, device='cpu'):
        x, lp = self.base.mean(context, device)
        x, logdet = self.transform.inverse(x, context)
        return x, lp - logdet

    def log_prob(self, x, context=None):
        x, logdet = self.transform.forward(x, context)
        lp = self.base.log_prob(x, context) + logdet
        return lp

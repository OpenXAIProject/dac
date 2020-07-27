import torch
import torch.nn.functional as F
import math

class MultivariateNormal(object):
    def __init__(self, dim):
        self.dim = dim
        self.dim_params = None

    def transform(self, raw):
        raise NotImplementedError

    def log_prob(self, X, params):
        raise NotImplementedError

    def stats(self, params):
        raise NotImplementedError

    def sample_params(self, shape, **kwargs):
        raise NotImplementedError

    def sample(self, params):
        raise NotImplementedError

class MultivariateNormalDiag(MultivariateNormal):
    def __init__(self, dim):
        super().__init__(dim)
        self.dim = dim
        self.dim_params = 2*dim

    def transform(self, raw):
        return torch.cat([raw[...,:self.dim],
            F.softplus(raw[...,self.dim:])], -1)

    def log_prob(self, X, params):
        mu = params[...,:self.dim]
        sigma = params[...,self.dim:]
        X = X.unsqueeze(-2)
        mu = mu.unsqueeze(-3)
        sigma = sigma.unsqueeze(-3)
        diff = X - mu
        ll = -0.5*math.log(2*math.pi) - sigma.log() - 0.5*(diff.pow(2)/sigma.pow(2))
        return ll.sum(-1)

    def stats(self, params):
        mu = params[...,:self.dim]
        sigma = params[...,self.dim:]
        I = torch.eye(self.dim)[(None,)*(len(sigma.shape)-1)].to(sigma.device)
        cov = sigma.pow(2).unsqueeze(-1) * I
        return mu, cov

    def sample_params(self, shape, device='cpu'):
        shape = torch.Size(shape) + torch.Size([self.dim])
        mu = 3.0 * torch.randn(shape).to(device)
        sigma = (math.log(0.25) + 0.1*torch.randn(shape)).exp().to(device)
        return torch.cat([mu, sigma], -1)

    def sample(self, params):
        mu = params[...,:self.dim]
        sigma = params[...,self.dim:]
        eps = torch.randn(mu.shape).to(mu.device)
        return mu + eps * sigma

class MultivariateNormalRankOne(MultivariateNormal):
    def __init__(self, dim):
        super().__init__(dim)
        self.dim = dim
        self.dim_params = 2*dim + 1

    def transform(self, raw):
        return torch.cat([raw[...,:self.dim],
            F.softplus(raw[...,self.dim:self.dim+1]),
            raw[...,self.dim+1:]], -1)

    def log_prob(self, X, params):
        mu = params[...,:self.dim]
        v = params[...,self.dim]
        W = params[...,self.dim+1:]
        X = X.unsqueeze(2)
        mu = mu.unsqueeze(1)
        v = v.unsqueeze(1)
        W = W.unsqueeze(1)
        dim = self.dim
        WtW = W.pow(2).sum(-1)
        log_norm = (
            0.5 * dim * math.log(2 * math.pi)
            + 0.5 * (1 + WtW / v).log()
            + 0.5 * dim * v.log()
        )
        diff = X - mu
        mdist = (diff.pow(2).sum(-1) - (W * diff).sum(-1).pow(2) / (v + WtW)) / v
        return -log_norm - 0.5 * mdist

    def stats(self, params):
        mu = params[...,:self.dim]
        v = params[...,self.dim]
        W = params[...,self.dim+1:]
        I = torch.eye(self.dim)[(None,) * len(v.shape)].to(v.device)
        cov = v[..., None, None] * I + torch.matmul(W.unsqueeze(-1), W.unsqueeze(-2))
        return mu, cov

    def sample_params(self, shape, device='cpu'):
        v = (math.log(0.1) + 0.1*torch.randn(shape)).to(device).exp()
        shape = torch.Size(shape) + torch.Size([self.dim])
        mu = 4.0 * torch.randn(shape).to(device)
        W = 0.25*torch.randn(shape).to(device)
        return torch.cat([mu, v.unsqueeze(-1), W], -1)

    def sample(self, params):
        mu = params[...,:self.dim]
        v = params[...,self.dim].unsqueeze(-1)
        W = params[...,self.dim+1:]

        eps_v = torch.randn(mu.shape).to(mu.device)
        eps_W = torch.randn(v.shape).to(v.device)

        return mu + eps_v * v.sqrt() + eps_W * W

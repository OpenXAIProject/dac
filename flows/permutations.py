import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.transform import Transform

class Flip(Transform):
    def forward(self, x, context=None):
        return x.flip(-1), 0

    def inverse(self, x, context=None):
        return x.flip(-1), 0

"""
copied & adapted from
https://github.com/y0ast/Glow-PyTorch
"""
class Invertible1x1Conv(Transform):
    def __init__(self, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        w_shape = [dim_inputs, dim_inputs]
        w_init = torch.qr(torch.randn(w_shape))[0]

        p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
        s = torch.diag(upper)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        upper = torch.triu(upper, 1)
        l_mask = torch.tril(torch.ones(w_shape), -1)
        eye = torch.eye(*w_shape)

        self.register_buffer('p', p)
        self.register_buffer('sign_s', sign_s)
        self.lower = nn.Parameter(lower)
        self.log_s = nn.Parameter(log_s)
        self.upper = nn.Parameter(upper)
        self.register_buffer('l_mask', l_mask)
        self.register_buffer('eye', eye)

    def build_weight(self, inverse=False):
        lower = self.lower * self.l_mask + self.eye
        u = self.upper * self.l_mask.transpose(0, 1).contiguous()
        u += torch.diag(self.sign_s * self.log_s.exp())
        if not inverse:
            return torch.matmul(self.p, torch.matmul(lower, u))
        else:
            u_inv = torch.inverse(u)
            l_inv = torch.inverse(lower)
            p_inv = torch.inverse(self.p)
            return torch.matmul(u_inv, torch.matmul(l_inv, p_inv))

    def forward(self, x, context=None):
        weight = self.build_weight(inverse=False)
        x = F.conv2d(x, weight[...,None,None])
        logdet = self.log_s.sum() * x.shape[-2]*x.shape[-1]
        logdet = logdet[(None,)*(x.dim()-3)]
        return x, logdet

    def inverse(self, x, context=None):
        weight = self.build_weight(inverse=True)
        x = F.conv2d(x, weight[...,None,None])
        logdet = -self.log_s.sum() * x.shape[-2]*x.shape[-1]
        logdet = logdet[(None,)*(x.dim()-3)]
        return x, logdet

class InvertibleLinear(Invertible1x1Conv):
    def forward(self, x, context=None):
        weight = self.build_weight(inverse=False)
        x = F.linear(x, weight, torch.zeros(self.dim_inputs, device=x.device))
        logdet = self.log_s.sum()[(None,)*(x.dim()-1)]
        return x, logdet

    def inverse(self, x, context=None):
        weight = self.build_weight(inverse=True)
        x = F.linear(x, weight, torch.zeros(self.dim_inputs, device=x.device))
        logdet = -self.log_s.sum()[(None,)*(x.dim()-1)]
        return x, logdet

if __name__ == '__main__':
    x = torch.rand(10, 128)
    layer = InvertibleLinear(128)
    y, logdet = layer(x)
    xr, _ = layer.inverse(y)

    print((x-xr).mean())
    print(x[0][:10])
    print(xr[0][:10])

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from utils.tensor import (chunk_two, cat_two,
        sum_except_batch as sumeb)
from flows.transform import Transform

"""
coupling transforms
copied from https://github.com/bayesiains/nsf
"""

class Coupling(Transform):
    def __init__(self, dim_inputs, net_fn, dim_context=0):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.dim_a = (dim_inputs + 1) // 2
        self.dim_b = dim_inputs // 2
        self.net = net_fn(self.dim_a, self.dim_params, dim_context)

    @property
    def dim_params(self):
        return self.dim_b

    def forward(self, x, context=None):
        xa, xb = chunk_two(x)
        xb, logdet = self.transform(xb, self.net(xa, context), inverse=False)
        return cat_two(xa, xb), logdet

    def inverse(self, x, context=None):
        xa, xb = chunk_two(x)
        xb, logdet = self.transform(xb, self.net(xa, context), inverse=True)
        return cat_two(xa, xb), logdet

class AffineCoupling(Coupling):
    @property
    def dim_params(self):
        return 2*self.dim_b

    def transform(self, xb, params, inverse=False):
        shift, scale = chunk_two(params)
        scale = torch.sigmoid(scale + 2.0) + 1e-3
        if not inverse:
            return scale*xb + shift, sumeb(scale.log())
        else:
            return (xb - shift) / scale, -sumeb(scale.log())

if __name__ == '__main__':

    class MLP(nn.Module):
        def __init__(self, dim_inputs, dim_outputs, dim_context=None):
            super().__init__()
            self.net = nn.Sequential(
                    nn.Linear(dim_inputs, dim_outputs),
                    nn.ELU(),
                    nn.Linear(dim_outputs, dim_outputs))
            if dim_context > 0:
                self.params_net = nn.Linear(dim_context, dim_outputs, bias=False)

        def forward(self, x, params=None):
            x = self.net(x)
            if params is not None and self.params_net is not None:
                x += self.params_net(params)
            return x

    dim_inputs = 5
    dim_context = 10
    net_fn = lambda d_in, d_out, d_con: MLP(d_in, d_out, d_con)
    layer = AffineCoupling(dim_inputs, net_fn, dim_context)
    x = torch.rand(10, dim_inputs)
    context = torch.rand(10, dim_context)

    y, _ = layer(x, context)
    xr, _ = layer.inverse(y, context)

    print((x-xr).mean())
    print(x[0])
    print(xr[0])

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tensor import chunk_two, sum_except_batch as sumeb
from flows.transform import Transform, Composite, Inverse
from flows.permutations import Flip, InvertibleLinear

"""
Largely adpated from
https://github.com/bayesiains/nsf
https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
"""

def build_mask(dim_inputs, dim_outputs, dim_flows, mask_type=None):
    """ mask_type: input | None | output
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(dim_inputs) % dim_flows
    else:
        in_degrees = torch.arange(dim_inputs) % (dim_flows - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(dim_outputs) % dim_flows - 1
    else:
        out_degrees = torch.arange(dim_outputs) % (dim_flows - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, mask):
        super().__init__()
        self.linear = nn.Linear(dim_inputs, dim_outputs)
        self.register_buffer('mask', mask)

    def forward(self, x):
        output = F.linear(x, self.linear.weight * self.mask, self.linear.bias)
        return output

class MADE(Transform):
    def __init__(self, dim_inputs, dim_hids, dim_context=0):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.linear = MaskedLinear(dim_inputs, dim_hids,
                build_mask(dim_inputs, dim_hids, dim_inputs, mask_type='input'))
        self.ctx_linear = nn.Linear(dim_context, dim_hids, bias=False) \
                if dim_context > 0 else None
        self.mlp = nn.Sequential(
                nn.ELU(),
                MaskedLinear(dim_hids, dim_hids,
                    build_mask(dim_hids, dim_hids, dim_inputs)),
                nn.ELU(),
                MaskedLinear(dim_hids, 2*dim_inputs,
                    build_mask(dim_hids, 2*dim_inputs, dim_inputs, mask_type='output')))

    @property
    def dim(self):
        return self.dim_inputs

    def get_params(self, x, context=None):
        h = self.linear(x)
        if context is not None and self.ctx_linear is not None:
            h = h + self.ctx_linear(context)
        h = self.mlp(h)
        shift, scale = chunk_two(h)
        scale = F.softplus(scale) + 1e-5
        return shift, scale

    def forward(self, x, context=None):
        shift, scale = self.get_params(x, context)
        return (x - shift)/scale, -sumeb(scale.log())

    def inverse(self, z, context=None):
        """
        inverse of MADE requires non-parallelizable computation,
        so better not use except special cases
        """
        x = torch.zeros_like(z)
        for i in range(self.dim):
            shift, scale = self.get_params(x, context)
            x[...,i] = z[...,i] * scale[...,i] + shift[...,i]
        return x, sumeb(scale.log())

class MAF(Composite):
    def __init__(self, dim_inputs, dim_hids, num_blocks,
            dim_context=0, inv_linear=False):
        transforms = []
        for _ in range(num_blocks):
            transforms.append(MADE(dim_inputs, dim_hids, dim_context))
            transforms.append(InvertibleLinear(dim_inputs) \
                    if inv_linear else Flip())
        super().__init__(transforms)

class IAF(Inverse):
    def __init__(self, dim_inputs, dim_hids, num_blocks,
            dim_context=0, inv_linear=False):
        super().__init__(MAF(dim_inputs, dim_hids, num_blocks,
            dim_context=dim_context, inv_linear=inv_linear))

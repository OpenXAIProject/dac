import torch
import torch.nn as nn
import torch.nn.functional as F
from flows.transform import Transform
from utils.tensor import chunk_two, sum_except_batch as sumeb

class ActNorm(Transform):
    def __init__(self, dim_inputs, dim_context=0):
        super().__init__()
        self.init = False
        self.dim_inputs = dim_inputs
        self.log_scale = nn.Parameter(torch.zeros(dim_inputs))
        self.shift = nn.Parameter(torch.zeros(dim_inputs))
        if dim_context > 0:
            self.linear = nn.Linear(dim_context, 2*dim_inputs)
            nn.init.uniform_(self.linear.weight, a=-1e-3, b=1e-3)
            nn.init.constant_(self.linear.bias, 0)
        else:
            self.linear = None

    def initialize(self, x):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).reshape(-1, self.dim_inputs)

        with torch.no_grad():
            std = x.std(0)
            mean = (x / std).mean(0)
            self.log_scale.data = -std.log()
            self.shift.data = -mean

        self.init = True

    def get_params(self, x, context=None):
        if x.dim() == 4:
            log_scale = self.log_scale.view(1, -1, 1, 1)
            shift = self.shift.view(1, -1, 1, 1)
            if context is not None and self.linear is not None:
                ctx_log_scale, ctx_shift = chunk_two(self.linear(context))
                B = x.shape[0]
                log_scale = log_scale + ctx_log_scale.view(B, -1, 1, 1)
                shift = shift + ctx_shift.view(B, -1, 1, 1)
            logdet = x.shape[-2]*x.shape[-1] * sumeb(log_scale)
        else:
            log_scale = self.log_scale.view(1, -1)
            shift = self.shift.view(1, -1)
            if context is not None and self.linear is not None:
                ctx_log_scale, ctx_shift = chunk_two(self.linear(context))
                B = x.shape[0]
                log_scale = log_scale + ctx_log_scale.view(B, -1)
                shift = shift + ctx_shift.view(B, -1)
            logdet = sumeb(log_scale)
        return log_scale, shift, logdet

    def forward(self, x, context=None):
        if self.training and not self.init:
            self.initialize(x)
        log_scale, shift, logdet = self.get_params(x, context=context)
        return log_scale.exp()*x + shift, logdet

    def inverse(self, x, context=None):
        log_scale, shift, logdet = self.get_params(x, context=context)
        return (x - shift)*(-log_scale).exp(), -logdet

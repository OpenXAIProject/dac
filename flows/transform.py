import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Largely adpated from
https://github.com/bayesiains/nsf
"""

class Transform(nn.Module):
    def forward(self, x, context=None):
        raise NotImplementedError

    def inverse(self, x, context=None):
        raise NotImplementedError

class Composite(Transform):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, context=None):
        logdet = 0
        for transform in self.transforms:
            x, logdet_ = transform.forward(x, context=context)
            logdet = logdet + logdet_
        return x, logdet

    def inverse(self, x, context=None):
        logdet = 0
        for transform in self.transforms[::-1]:
            x, logdet_ = transform.inverse(x, context=context)
            logdet = logdet + logdet_
        return x, logdet

class Inverse(Transform):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x, context=None):
        return self.transform.inverse(x, context=context)

    def inverse(self, x, context=None):
        return self.transform.forward(x, context=context)

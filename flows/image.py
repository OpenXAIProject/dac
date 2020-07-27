import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flows.transform import Transform
from utils.tensor import chunk_two, cat_two

class Squeeze(Transform):
    """
    Implementation of squeeze layer
    adapted from
    https://github.com/y0ast/Glow-PyTorch
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def output_shape(self, shape):
        C, H, W = shape
        return (C*self.factor**2, H//self.factor, W//self.factor)

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        factor = self.factor
        assert H % factor == 0 and W % factor == 0
        x = x.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x, 0

    def inverse(self, x, context=None):
        B, C, H, W = x.shape
        factor = self.factor
        factor2 = factor**2
        assert C % factor2 == 0
        x = x.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // factor2, H * factor, W * factor)
        return x, 0

class MultiscaleComposite(Transform):
    """A multiscale composite transform as described in the RealNVP paper.

    Copied & adapted from https://github.com/bayesiains/nsf

    Splits the outputs along the given dimension after every transform, outputs one half, and
    passes the other half to further transforms. No splitting is done before the last transform.

    Note: Inputs could be of arbitrary shape, but outputs will always be flattened.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self):
        super().__init__()
        self.transforms = nn.ModuleList()
        self.output_shapes = []

    def append(self, transform, output_shape, last=False):
        self.transforms.append(transform)
        if not last:
            C, H, W = output_shape
            self.output_shapes.append(((C + 1)//2, H, W))
            return (C//2, H, W)
        else:
            self.output_shapes.append(output_shape)
            return output_shape

    def forward(self, x, context=None):
        B = x.shape[0]
        logdet = 0
        finished = []
        for i, transform in enumerate(self.transforms[:-1]):
            x, logdet_ = transform(x, context)
            logdet = logdet + logdet_
            x_, x = chunk_two(x)
            assert x_.shape[1:] == self.output_shapes[i]
            finished.append(x_.view(B, -1))

        x, logdet_ = self.transforms[-1](x, context)
        logdet = logdet + logdet_
        finished.append(x.view(B, -1))
        x = torch.cat(finished, -1)
        return x, logdet

    def inverse(self, x, context=None):
        split_x = []
        off = 0
        for shape in self.output_shapes:
            C, H, W = shape
            split_x.append(x[:, off:off+C*H*W].view(-1, C, H, W))
            off += C*H*W

        x, logdet = self.transforms[-1].inverse(split_x[-1], context)
        for x_, transform in zip(split_x[-2::-1], self.transforms[-2::-1]):
            x, logdet_ = transform.inverse(cat_two(x_, x), context)
            logdet = logdet + logdet_

        return x, logdet

# take [0, 1] float image, add uniform noise and shift
class Dequantize(Transform):
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.num_bins = 2**num_bits

    def forward(self, x, context=None):
        _, C, H, W = x.shape
        # undo torch.ToTensor and add uniform noise
        x = (255.*x + torch.rand_like(x)) / self.num_bins - 0.5
        return x, -math.log(self.num_bins)*C*H*W

    def inverse(self, x, context=None):
        _, C, H, W = x.shape
        x = torch.floor(self.num_bins*(x + 0.5)) / 255.
        x = x.clamp(0., 1.)
        return x, math.log(self.num_bins)*C*H*W

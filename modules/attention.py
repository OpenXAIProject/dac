import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_X, dim_Y, dim, num_heads=4, ln=False, p=None):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_X, dim)
        self.fc_k = nn.Linear(dim_Y, dim)
        self.fc_v = nn.Linear(dim_Y, dim)
        self.fc_o = nn.Linear(dim, dim)

        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.dropout1 = nn.Dropout(p=p) if p is not None else nn.Identity()
        self.dropout2 = nn.Dropout(p=p) if p is not None else nn.Identity()

    def forward(self, X, Y, mask=None):
        Q, K, V = self.fc_q(X), self.fc_k(Y), self.fc_v(Y)
        Q_ = torch.cat(Q.chunk(self.num_heads, -1), 0)
        K_ = torch.cat(K.chunk(self.num_heads, -1), 0)
        V_ = torch.cat(V.chunk(self.num_heads, -1), 0)

        A_logits = (Q_ @ K_.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
        if mask is not None:
            mask = torch.stack([mask]*Q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, 0)
            A_logits.masked_fill_(mask, -float('inf'))
            A = torch.softmax(A_logits, -1)
            # to prevent underflow due to no attention
            A.masked_fill_(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)

        attn = torch.cat((A @ V_).chunk(self.num_heads, 0), -1)
        O = self.ln1(Q + self.dropout1(attn))
        O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
        return O

class SAB(nn.Module):
    def __init__(self, dim_X, dim, **kwargs):
        super().__init__()
        self.mab = MAB(dim_X, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)

class StackedSAB(nn.Module):
    def __init__(self, dim_X, dim, num_blocks, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
                [SAB(dim_X, dim, **kwargs)] + \
                [SAB(dim, dim, **kwargs)]*(num_blocks-1))

    def forward(self, X, mask=None):
        for sab in self.blocks:
            X = sab(X, mask=mask)
        return X

class PMA(nn.Module):
    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(num_inds, dim))
        nn.init.xavier_uniform_(self.I)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        I = self.I if X.dim() == 2 else self.I.repeat(X.shape[0], 1, 1)
        return self.mab(I, X, mask=mask)

class ISAB(nn.Module):
    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.pma = PMA(dim_X, dim, num_inds, **kwargs)
        self.mab = MAB(dim_X, dim, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(X, self.pma(X, mask=mask))

class StackedISAB(nn.Module):
    def __init__(self, dim_X, dim, num_inds, num_blocks, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
                [ISAB(dim_X, dim, num_inds, **kwargs)] + \
                [ISAB(dim, dim, num_inds, **kwargs)]*(num_blocks-1))

    def forward(self, X, mask=None):
        for isab in self.blocks:
            X = isab(X, mask=mask)
        return X

class aPMA(nn.Module):
    def __init__(self, dim_X, dim, **kwargs):
        super().__init__()
        self.I0 = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.I0)
        self.pma = PMA(dim, dim, 1, **kwargs)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, num_iters):
        I = self.I0
        for i in range(1, num_iters):
            I = torch.cat([I, self.pma(I)], 1)
        return self.mab(I.repeat(X.shape[0], 1, 1), X)

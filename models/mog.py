import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.base import sample_anchors

from utils.paths import benchmarks_path
from utils.misc import add_args
from utils.plots import scatter_mog, draw_ellipse, scatter

from modules.attention import StackedISAB, PMA, MAB
from data.mvn import MultivariateNormalDiag
from data.mog import sample_mog

from models.base import MinFilteringModel, AnchoredFilteringModel

parser = argparse.ArgumentParser()

# for training
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--num_steps', type=int, default=20000)

parser.add_argument('--loss_type', type=str, choices=['min', 'anc'], default='min')
parser.add_argument('--filtering_benchmark', type=str, default=None)
parser.add_argument('--clustering_benchmark', type=str, default=None)

# for visualization
parser.add_argument('--vB', type=int, default=10)
parser.add_argument('--vN', type=int, default=1000)
parser.add_argument('--vK', type=int, default=4)

sub_args, _ = parser.parse_known_args()

class MinFilteringNetwork(nn.Module):
    def __init__(self, mvn, dim_hids=128, num_inds=32):
        super().__init__()
        self.mvn = mvn
        self.isab1 = StackedISAB(mvn.dim, dim_hids, num_inds, 4)
        self.pma = PMA(dim_hids, dim_hids, 1)
        self.fc1 = nn.Linear(dim_hids, mvn.dim_params)

        self.mab = MAB(dim_hids, dim_hids, dim_hids)
        self.isab2 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc2 = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None):
        # encode data
        H_X = self.isab1(X, mask=mask)

        # extract params of clusters including anchors
        H_theta = self.pma(H_X, mask=mask)
        theta = self.mvn.transform(self.fc1(H_theta))
        ll = self.mvn.log_prob(X, theta).squeeze(-1)
        theta = theta.squeeze(-2)

        # extract membership vector logits
        H_m = self.mab(H_X, H_theta)
        H_m = self.isab2(H_m, mask=mask)
        logits = self.fc2(H_m).squeeze(-1)

        return {'theta':theta, 'll':ll, 'logits':logits}

class AnchoredFilteringNetwork(nn.Module):
    def __init__(self, mvn, dim_hids=128, num_inds=32):
        super().__init__()
        self.mvn = mvn
        self.mab1 = MAB(mvn.dim, mvn.dim, dim_hids)
        self.isab1 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.pma = PMA(dim_hids, dim_hids, 1)
        self.fc1 = nn.Linear(dim_hids, mvn.dim_params)

        self.mab2 = MAB(dim_hids, dim_hids, dim_hids)
        self.isab2 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc2 = nn.Linear(dim_hids, 1)

    def forward(self, X, anc_idxs, mask=None):
        # encode data
        xa = X[torch.arange(X.shape[0]), anc_idxs].unsqueeze(-2)
        H_Xa = self.isab1(self.mab1(X, xa), mask=mask)

        # extract params of clusters including anchors
        H_theta = self.pma(H_Xa, mask=mask)
        theta = self.mvn.transform(self.fc1(H_theta))
        ll = self.mvn.log_prob(X, theta).squeeze(-1)
        theta = theta.squeeze(-2)

        # extract membership vector logits
        H_m = self.mab2(H_Xa, H_theta)
        H_m = self.isab2(H_m, mask=mask)
        logits = self.fc2(H_m).squeeze(-1)

        return {'theta':theta, 'll':ll, 'logits':logits}

if sub_args.loss_type == 'min':
    Parent = MinFilteringModel
    Net = MinFilteringNetwork
else:
    Parent = AnchoredFilteringModel
    Net = AnchoredFilteringNetwork

class Model(Parent):
    def __init__(self, args):
        super().__init__(args)

        self.filtering_benchmark = os.path.join(benchmarks_path, 'mog_10_1000_4.tar') \
                if self.filtering_benchmark is None \
                else os.path.join(benchmarks_path, self.filtering_benchmark)

        self.clustering_benchmark = os.path.join(benchmarks_path, 'mog_10_3000_12.tar') \
                if self.clustering_benchmark is None \
                else os.path.join(benchmarks_path, self.clustering_benchmark)

        self.net = Net(MultivariateNormalDiag(2))

        self.train_metrics = ['ll', 'bcent']
        self.test_metrics = ['ll', 'bcent']

    def gen_benchmarks(self, force=False):
        if not os.path.isfile(self.filtering_benchmark) or force:
            print('generating benchmark {}...'.format(self.filtering_benchmark))
            bench = []
            for _ in range(100):
                bench.append(sample_mog(10, 1000, 4,
                    rand_N=True, rand_K=True, return_ll=True))
            torch.save(bench, self.filtering_benchmark)
        if not os.path.isfile(self.clustering_benchmark) or force:
            print('generating benchmark {}...'.format(self.clustering_benchmark))
            bench = []
            for _ in range(100):
                bench.append(sample_mog(10, 3000, 12,
                    rand_N=True, rand_K=True, return_ll=True))
            torch.save(bench, self.clustering_benchmark)

    def sample(self, B, N, K, **kwargs):
        return sample_mog(B, N, K, device='cuda', **kwargs)

    def plot_clustering(self, X, results):
        labels = results.get('labels')
        theta = results.get('theta')
        B = X.shape[0]
        mu, cov = self.net.mvn.stats(theta)
        fig, axes = plt.subplots(min(B, 2), max(B//2, 1),
                figsize=(2.5*B if B > 1 else 10, 10))
        axes = [axes] if B == 1 else axes.flatten()
        for b, ax in enumerate(axes):
            scatter_mog(X[b], labels[b], mu[b], cov[b], ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])

    def plot_filtering(self, batch):
        X = batch['X']
        B = X.shape[0]
        with torch.no_grad():
            outputs = self.compute_loss(batch, train=False)
        mu, cov = self.net.mvn.stats(outputs['theta'])
        labels = (outputs['logits'] > 0.0).long()
        fig, axes = plt.subplots(min(B, 2), max(B//2, 1),
                figsize=(2.5*B if B > 1 else 10, 10))
        axes = [axes] if B == 1 else axes.flatten()
        for b, ax in enumerate(axes):
            scatter(X[b], labels=labels[b], ax=ax)
            draw_ellipse(mu[b], cov[b], ax=ax)

def load(args):
    add_args(args, sub_args)
    return Model(args)

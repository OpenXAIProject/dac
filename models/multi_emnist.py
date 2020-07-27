import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid

from torch.nn.utils import weight_norm

from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path
from utils.tensor import to_numpy

from data.base import sample_anchors
from data.multi_emnist import ClusteredMultiEMNIST, sample_idxs, \
        get_train_loader, get_test_loader

from modules.attention import StackedISAB, PMA, MAB
from modules.misc import Flatten, View

from flows.autoregressive import MAF
from flows.distributions import FlowDistribution, Normal, Bernoulli

from models.base import AnchoredFilteringModel, MinFilteringModel

parser = argparse.ArgumentParser()

# for training
parser.add_argument('--B', type=int, default=10)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--num_steps', type=int, default=20000)

parser.add_argument('--filtering_benchmark', type=str, default=None)
parser.add_argument('--clustering_benchmark', type=str, default=None)

parser.add_argument('--vB', type=int, default=1)
parser.add_argument('--vN', type=int, default=100)
parser.add_argument('--vK', type=int, default=4)

sub_args, _ = parser.parse_known_args()

class FilteringNetwork(nn.Module):
    def __init__(self, num_filters=32, dim_lats=128,
            dim_hids=256, dim_context=256, num_inds=32):
        super().__init__()

        C = num_filters
        self.enc = nn.Sequential(
                nn.Conv2d(3, C, 3, stride=2),
                nn.BatchNorm2d(C),
                nn.ReLU(),
                nn.Conv2d(C, 2*C, 3, stride=2),
                nn.BatchNorm2d(2*C),
                nn.ReLU(),
                nn.Conv2d(2*C, 4*C, 3),
                Flatten())

        self.isab1 = StackedISAB(4*C*4*4, dim_hids, num_inds, 4)
        self.pma = PMA(dim_hids, dim_hids, 1)
        self.fc1 = nn.Linear(dim_hids, dim_context)

        self.posterior = Normal(dim_lats, use_context=True,
                context_enc=nn.Linear(4*C*4*4 + dim_context, 2*dim_lats))
        self.prior = FlowDistribution(
                MAF(dim_lats, dim_hids, 6, dim_context=dim_context, inv_linear=True),
                Normal(dim_lats))

        self.dec = nn.Sequential(
                nn.Linear(dim_lats + dim_context, 4*C*4*4),
                nn.ReLU(),
                View(-1, 4*C, 4, 4),
                nn.ConvTranspose2d(4*C, 2*C, 3, stride=2, padding=1),
                nn.BatchNorm2d(2*C),
                nn.ReLU(),
                nn.ConvTranspose2d(2*C, C, 3, stride=2, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(),
                nn.ConvTranspose2d(C, 3, 3, stride=2, output_padding=1),
                View(-1, 3, 28, 28))
        self.likel = Bernoulli((3, 28, 28), use_context=True)

        self.mab = MAB(dim_hids, dim_hids, dim_hids)
        self.isab2 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc2 = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None, return_z=False):
        B, N, C, H, W = X.shape
        x = X.view(B*N, C, H, W)
        h_enc = self.enc(x)

        H_X = self.isab1(h_enc.view(B, N, -1), mask=mask)
        H_theta = self.pma(H_X, mask=mask)
        theta = self.fc1(H_theta)
        theta_ = theta.repeat(1, N, 1).view(B*N, -1)

        z, logq = self.posterior.sample(context=torch.cat([h_enc, theta_], -1))
        logp = self.prior.log_prob(z, context=theta_)
        kld = (logq - logp).view(B, N)

        h_dec = self.dec(torch.cat([z, theta_], -1))
        ll = self.likel.log_prob(x, context=h_dec).view(B, N) - kld
        ll /= H*W

        H_dec = self.mab(H_X, H_theta)
        logits = self.fc2(self.isab2(H_dec, mask=mask)).squeeze(-1)

        outputs = {'ll':ll, 'theta':theta, 'logits':logits}
        if return_z:
            outputs['z'] = z
        return outputs

class Model(MinFilteringModel):
    def __init__(self, args):
        super().__init__(args)

        self.filtering_benchmark = os.path.join(benchmarks_path, 'memnist_10_100_4.tar') \
                if self.filtering_benchmark is None \
                else os.path.join(benchmarks_path, self.filtering_benchmark)

        self.clustering_benchmark = os.path.join(benchmarks_path, 'memnist_10_300_12.tar') \
                if self.clustering_benchmark is None \
                else os.path.join(benchmarks_path, self.clustering_benchmark)

        self.net = FilteringNetwork()
        self.train_metrics = ['ll', 'bcent']
        self.test_metrics = ['ll', 'bcent']

    def sample(self, B, N, K, **kwargs):
        dataset = ClusteredMultiEMNIST(train=False)
        batch = sample_idxs(dataset.idx_to_class, B, N, K, **kwargs)
        return dataset[batch]

    def build_train_loader(self):
        self.train_loader = get_train_loader(self.B, self.N, self.K, self.num_steps,
                rand_N=True, rand_K=True)

    def build_test_loader(self, filename=None):
        filename = self.filtering_benchmark if filename is None else filename
        self.test_loader = get_test_loader(filename)

    def gen_benchmarks(self, force=False):
        if not os.path.isfile(self.filtering_benchmark) or force:
            print('generating benchmark {}...'.format(self.filtering_benchmark))
            idx_to_class = ClusteredMultiEMNIST(train=False).idx_to_class
            bench = [sample_idxs(idx_to_class, 10, 100, 4, rand_N=True, rand_K=True) \
                    for _ in range(100)]
            torch.save(bench, self.filtering_benchmark)

            print('generating benchmark {}...'.format(self.clustering_benchmark))
            bench = [sample_idxs(idx_to_class, 10, 300, 12, rand_N=True, rand_K=True) \
                    for _ in range(100)]
            torch.save(bench, self.clustering_benchmark)

    def combine_digits(self, X):
        B, N, C, H, W = X.shape
        cX = torch.zeros(B, N, 1, 2*H, 2*W)
        cX[:,:,0,:H,:W] = X[:,:,0,:,:]
        cX[:,:,0,:H,W:] = X[:,:,1,:,:]
        cX[:,:,0,H:,:W] = X[:,:,2,:,:]
        return cX

    def plot_clustering(self, X, results):
        X = self.combine_digits(X)[0]
        labels = results['labels'][0]
        ulabels = torch.unique(labels)
        K = len(ulabels)
        fig, axes = plt.subplots(1, K, figsize=(50, 50))
        for k, l in enumerate(ulabels):
            Xk = X[labels==l]
            Xk = Xk[: Xk.shape[0] - Xk.shape[0] % 4]
            I = to_numpy(make_grid(1-Xk, nrow=4, pad_value=0)).transpose(1, 2, 0)
            axes[k].set_title('cluster {}'.format(k+1), fontsize=100)
            axes[k].imshow(I)
            axes[k].axis('off')
        plt.tight_layout()

    def plot_filtering(self, batch):
        X = batch['X'].cuda()
        B, N, C, H, W = X.shape
        net = self.net
        net.eval()
        with torch.no_grad():
            outputs = net(X, return_z=True)
            theta = outputs['theta']
            theta_ = theta.repeat(1, N, 1).view(B*N, -1)
            labels = (outputs['logits'] > 0.0).long()

            # conditional generation
            z, _ = net.prior.sample(B*N, device='cuda', context=theta_)
            h_dec = net.dec(torch.cat([z, theta_], -1))
            gX, _ = net.likel.sample(context=h_dec)
            gX = gX.view(B, N, C, H, W)

            z = outputs['z']
            h_dec = net.dec(torch.cat([z, theta_], -1))
            rX, _ = net.likel.sample(context=h_dec)
            rX = rX.view(B, N, C, H, W)

        fig, axes = plt.subplots(1, 2, figsize=(40, 40))
        X = self.combine_digits(X)[0]
        labels = labels[0]

        X1 = X[labels==1]
        X1 = X1[: X1.shape[0] - X1.shape[0] % 8]
        I = to_numpy(make_grid(1-X1, nrow=8, pad_value=0)).transpose(1, 2, 0)
        axes[0].imshow(I)
        axes[0].set_title('Filtered out images', fontsize=60, pad=20)
        axes[0].axis('off')

        X0 = X[labels==0]
        X0 = X0[: X0.shape[0] - X0.shape[0] % 8]
        I = to_numpy(make_grid(1-X0, nrow=8, pad_value=0)).transpose(1, 2, 0)
        axes[1].imshow(I)
        axes[1].set_title('Remaining images', fontsize=60, pad=20)
        axes[1].axis('off')
        plt.tight_layout()
        #plt.savefig('figures/emnist_filtering.png', bbox_inches='tight')

        gX = self.combine_digits(gX)[0][:32]
        plt.figure()
        I = to_numpy(make_grid(1-gX, nrow=8, pad_value=0)).transpose(1, 2, 0)
        plt.imshow(I)
        plt.title('Generated images', fontsize=15, pad=5)
        plt.axis('off')
        #plt.savefig('figures/emnist_gen.png', bbox_inches='tight')

        fig, axes = plt.subplots(1, 2, figsize=(40, 40))
        rX = self.combine_digits(rX)[0]
        X1 = rX[labels==1]
        X1 = X1[: X1.shape[0] - X1.shape[0] % 8]
        I = to_numpy(make_grid(1-X1, nrow=8, pad_value=0)).transpose(1, 2, 0)
        axes[0].imshow(I)
        axes[0].set_title('Reconstructions of filtered out images', fontsize=60, pad=20)
        axes[0].axis('off')

        X0 = rX[labels==0]
        X0 = X0[: X0.shape[0] - X0.shape[0] % 8]
        I = to_numpy(make_grid(1-X0, nrow=8, pad_value=0)).transpose(1, 2, 0)
        axes[1].imshow(I)
        axes[1].set_title('Reconstructions of remaining images', fontsize=60, pad=20)
        axes[1].axis('off')
        plt.tight_layout()
        #plt.savefig('figures/emnist_recon.png', bbox_inches='tight')

def load(args):
    add_args(args, sub_args)
    return Model(args)

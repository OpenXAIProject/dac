import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
import torchvision.transforms as tvt

from data.base import sample_partitions
from utils.paths import datasets_path

NUM_DIGITS = 3
NUM_CLASSES = 47
NUM_CLUSTERS = NUM_CLASSES**NUM_DIGITS
BASE = (np.cumprod(NUM_CLASSES*np.ones(NUM_DIGITS, dtype=np.long))//NUM_CLASSES)[::-1]

def encode(classes):
    return (classes * BASE).sum()

def decode(cluster):
    classes = []
    for b in BASE:
        classes.append(cluster // b)
        cluster = cluster % b
    return classes

def sample_idxs(idx_to_class, B, N, K, rand_N=True, rand_K=True, train=True):
    N = np.random.randint(int(0.3*N), N) if rand_N else N
    labels = sample_partitions(B, N, K, rand_K=rand_K, device='cpu', alpha=5.0)

    idxs = torch.zeros(B, N, NUM_DIGITS,  dtype=torch.long)
    abs_labels = torch.zeros(B, N, dtype=torch.long)

    cluster_pool = np.arange(NUM_CLUSTERS//2) if train \
            else np.arange(NUM_CLUSTERS//2, NUM_CLUSTERS)
    for b in range(B):
        clusters = np.random.permutation(cluster_pool)[:K]
        for i, cls in enumerate(clusters):
            if (labels[b] == i).int().sum() > 0:
                members = (labels[b] == i).nonzero().view(-1)
                abs_labels[b, members] = np.long(cls)

                classes = decode(cls)
                for d, c in enumerate(classes):
                    idx_pool = idx_to_class[c]
                    idx_pool = idx_pool[torch.randperm(len(idx_pool))]
                    n_repeat = len(members) // len(idx_pool) + 1
                    idxs[b, members, d] = torch.cat([idx_pool]*n_repeat)[:len(members)]

    idxs = idxs.view(B, -1)
    oh_labels = F.one_hot(labels, K)
    return {'idxs':idxs, 'oh_labels':oh_labels, 'abs_labels':abs_labels}

class ClusteredMultiEMNIST(object):
    def __init__(self, train=True):

        rotate = lambda x: x.transpose(-1, -2)

        self.dataset = EMNIST(datasets_path, train=train, split='balanced',
                transform=tvt.Compose([tvt.ToTensor(),
                    tvt.Lambda(rotate),
                    tvt.Lambda(torch.bernoulli)]))
        self.idx_to_class = {l:(self.dataset.targets==l).nonzero().squeeze() \
                for l in range(NUM_CLASSES)}

    def __getitem__(self, batch):
        idxs = batch.pop('idxs')
        B = idxs.shape[0]
        X = (self.dataset[i] for i in idxs.flatten())
        X = torch.stack([x for x, _, in X])
        batch['X'] = X.view(B, -1, NUM_DIGITS, 28, 28)
        return batch

class TrainSampler(object):
    def __init__(self, sample_fn, num_steps):
        self.num_steps = num_steps
        self.sample_fn = sample_fn

    def __iter__(self):
        for _ in range(self.num_steps):
            yield [self.sample_fn()]

    def __len__(self):
        return self.num_steps

class TestSampler(object):
    def __init__(self, filename):
        self.batches = [[batch] for batch in torch.load(filename)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def get_train_loader(B, N, K, num_steps, **kwargs):
    dataset = ClusteredMultiEMNIST(train=True)
    sample_fn = lambda : sample_idxs(dataset.idx_to_class, B, N, K, **kwargs)
    return DataLoader(dataset, batch_sampler=TrainSampler(sample_fn, num_steps),
            num_workers=4, collate_fn=lambda x: x[0])

def get_test_loader(filename):
    dataset = ClusteredMultiEMNIST(train=False)
    sampler = TestSampler(filename)
    return DataLoader(dataset, batch_sampler=TestSampler(filename),
            num_workers=4, collate_fn=lambda x: x[0])

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    loader = get_train_loader(10, 64, 4, 1, rand_K=False, rand_N=False)
    batch = next(iter(loader))
    print(batch['oh_labels'][0])

    for ll in batch['abs_labels'][0]:
        print(decode(ll))

    X = batch['X'][0]

    plt.subplot(131)
    I = make_grid(X[:, 0:1]).numpy().transpose(1,2,0)
    plt.imshow(I)

    plt.subplot(132)
    I = make_grid(X[:, 1:2]).numpy().transpose(1,2,0)
    plt.imshow(I)

    plt.subplot(133)
    I = make_grid(X[:, 2:]).numpy().transpose(1,2,0)
    plt.imshow(I)

    plt.show()

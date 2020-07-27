import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from data.base import sample_partitions

def sample_idxs(idx_to_class, B, N, K, rand_N=True, rand_K=True):
    N = np.random.randint(int(0.3*N), N) if rand_N else N
    labels = sample_partitions(B, N, K, rand_K=rand_K, device='cpu', alpha=5.0)

    idxs = torch.zeros(B, N, dtype=torch.long)
    abs_labels = torch.zeros(B, N, dtype=torch.long)

    classes_pool = list(idx_to_class.keys())
    for b in range(B):
        classes = np.random.permutation(classes_pool)[:K]
        for i, c in enumerate(classes):
            if (labels[b] == i).int().sum() > 0:
                members = (labels[b] == i).nonzero().view(-1)
                idx_pool = idx_to_class[c]
                idx_pool = idx_pool[torch.randperm(len(idx_pool))]
                n_repeat = len(members) // len(idx_pool) + 1
                idxs[b, members] = torch.cat([idx_pool]*n_repeat)[:len(members)]
                abs_labels[b, members] = np.long(c)

    oh_labels = F.one_hot(labels, K)
    return {'idxs':idxs, 'oh_labels':oh_labels, 'abs_labels':abs_labels}

class ClusteredDataset(object):
    def __init__(self, dataset, classes=None):
        self.dataset = dataset
        if not type(self.dataset.targets) == torch.Tensor:
            self.dataset.targets = torch.Tensor(self.dataset.targets)

        if classes is None:
            self.classes = torch.unique(self.dataset.targets).numpy()
        else:
            self.classes = np.array(classes)

        self.num_classes = len(self.classes)

        self.idx_to_class = {l:(dataset.targets==l).nonzero().squeeze() \
                for l in self.classes}

    def __getitem__(self, batch):
        idxs = batch.pop('idxs')
        X = (self.dataset[i] for i in idxs.flatten())
        X = torch.stack([x for x, _, in X])
        _, *shape = X.shape
        batch['X'] = X.reshape(*idxs.shape, *shape)
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

def get_train_loader(dataset, B, N, K, num_steps, classes=None, **kwargs):
    dataset = ClusteredDataset(dataset, classes=classes)
    sample_fn = lambda : sample_idxs(dataset.idx_to_class, B, N, K, **kwargs)
    return DataLoader(dataset, batch_sampler=TrainSampler(sample_fn, num_steps),
            num_workers=4, collate_fn=lambda x: x[0])

def get_test_loader(dataset, filename, classes=None):
    dataset = ClusteredDataset(dataset, classes=classes)
    sampler = TestSampler(filename)
    return DataLoader(dataset, batch_sampler=TestSampler(filename),
            num_workers=4, collate_fn=lambda x: x[0])

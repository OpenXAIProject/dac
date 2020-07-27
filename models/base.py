import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.losses import compute_min_filtering_loss, \
        compute_anchored_filtering_loss

from data.base import sample_anchors
import data.clustered_dataset as cds

from utils.tensor import to_numpy

class BaseModel(object):
    def __init__(self, args):
        for key, value in args.__dict__.items():
            setattr(self, key, value)

        # self.net = ...
        # self.lr = ...
        # self.num_steps = ...
        # self.clip = ...

        # self.train_metrics = [...]
        # self.test_metrics = [...]

        # self.filtering_benchmark = ...
        # self.B = ...
        # self.N = ...
        # self.K = ...

    def cuda(self):
        self.net.cuda()

    def sample(self, B, N, K):
        raise NotImplementedError

    def build_train_loader(self):
        def train_loader():
            for _ in range(self.num_steps):
                yield self.sample(self.B, self.N, self.K)
        self.train_loader = train_loader()

    def build_test_loader(self, filename=None):
        self.test_loader = torch.load(self.filtering_benchmark if filename is None else filename)

    def build_optimizer(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, T_max=self.num_steps)

    def compute_loss(self, batch, train=True):
        raise NotImplementedError

    def get_lr_string(self):
        return 'lr {:.2e}'.format(self.optimizer.param_groups[0]['lr'])

    def train_batch(self, batch, accm):
        self.net.train()
        self.optimizer.zero_grad()
        outputs = self.compute_loss(batch)
        outputs['loss'].backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.optimizer.step()
        self.scheduler.step()
        for key in self.train_metrics:
            accm.update(key, outputs[key])

    def test(self, accm):
        self.net.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                outputs = self.compute_loss(batch, train=False)
                for key in self.test_metrics:
                    accm.update(key, outputs[key])

    def save(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))

    def cluster(self, X, max_iter=50, verbose=True):
        raise NotImplementedError

class MinFilteringModel(BaseModel):
    def compute_loss(self, batch, train=True):
        outputs = self.net(batch['X'].cuda())
        compute_min_filtering_loss(outputs, batch['oh_labels'].cuda().float())
        return outputs

    def cluster(self, X, max_iter=50, verbose=True):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(X)
            theta = [outputs.get('theta', None)]
            ll = [outputs.get('ll', None)]
            labels = torch.zeros_like(outputs['logits']).long()
            mask = outputs['logits'] > 0.0
            done = mask.sum(-1) == N
            for i in range(1, max_iter):
                outputs = self.net(X, mask=mask)
                theta.append(outputs.get('theta', None))
                ll.append(outputs.get('ll', None))

                ind = outputs['logits'] > 0.0
                labels[ind*mask.bitwise_not()] = i
                mask[ind] = True

                num_processed = mask.sum(-1)
                done = num_processed == N
                if verbose:
                    print(to_numpy(num_processed))
                if done.sum() == B:
                    break
            if ll[0] is not None:
                ll = torch.stack(ll, -1)
                theta = torch.stack(theta, -2)

                pi = F.one_hot(labels, ll.shape[-1]).float()
                pi = pi.sum(1, keepdim=True) / pi.shape[1]
                ll = ll + (pi + 1e-10).log()
                ll = ll.logsumexp(-1).mean()

                return {'theta':theta, 'll':ll, 'labels':labels}
            else:
                return {'labels':labels}

class AnchoredFilteringModel(BaseModel):
    def compute_loss(self, batch, train=True):
        X = batch['X'].cuda()
        oh_labels = batch['oh_labels'].cuda().float()
        anc_idxs = sample_anchors(X.shape[0], X.shape[1])
        outputs = self.net(X, anc_idxs)
        compute_anchored_filtering_loss(outputs, anc_idxs, oh_labels)
        return outputs

    def cluster(self, X, max_iter=50, verbose=True):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()
        with torch.no_grad():
            anc_idxs = sample_anchors(B, N)
            outputs = self.net(X, anc_idxs)
            theta = [outputs.get('theta', None)]
            ll = [outputs.get('ll', None)]
            labels = torch.zeros_like(outputs['logits']).long()
            mask = outputs['logits'] > 0.0
            done = mask.sum(-1) == N
            for i in range(1, max_iter):
                anc_idxs = sample_anchors(B, N, mask=mask)
                outputs = self.net(X, anc_idxs, mask=mask)
                theta.append(outputs.get('theta', None))
                ll.append(outputs.get('ll', None))

                ind = outputs['logits'] > 0.0
                labels[ind*mask.bitwise_not()] = i
                mask[ind] = True

                num_processed = mask.sum(-1)
                done = num_processed == N
                if verbose:
                    print(to_numpy(num_processed))
                if done.sum() == B:
                    break

            if ll[0] is not None:
                ll = torch.stack(ll, -1)
                theta = torch.stack(theta, -2)

                pi = F.one_hot(labels, ll.shape[-1]).float()
                pi = pi.sum(1, keepdim=True) / pi.shape[1]
                ll = ll + (pi + 1e-10).log()
                ll = ll.logsumexp(-1).mean()

                return {'theta':theta, 'll':ll, 'labels':labels}
            else:
                return {'labels':labels}

# models used for real-world datasets
class RealMinFilteringModel(MinFilteringModel):
    def __init__(self, args):
        super().__init__(args)
        self.train_metrics = ['bcent']
        self.test_metrics = ['bcent']

    def get_dataset(self, train=True):
        raise NotImplementedError

    def get_classes(self, train=True):
        return None

    def build_train_loader(self):
        self.train_loader = cds.get_train_loader(self.get_dataset(),
                self.B, self.N, self.K, self.num_steps,
                classes=self.get_classes())

    def build_test_loader(self, filename=None):
        self.test_loader = cds.get_test_loader(self.get_dataset(False),
                (self.filtering_benchmark if filename is None else filename),
                classes=self.get_classes(False))

    def gen_benchmarks(self, force=False):
        if not os.path.isfile(self.filtering_benchmark) or force:
            print('generating benchmark {}...'.format(self.filtering_benchmark))

            dataset = cds.ClusteredDataset(self.get_dataset(False),
                    classes=self.get_classes(False))
            bench = [cds.sample_idxs(dataset.idx_to_class,
                10, 100, 4, rand_N=True, rand_K=True) for _ in range(100)]
            torch.save(bench, self.filtering_benchmark)

            print('generating benchmark {}...'.format(self.clustering_benchmark))
            bench = [cds.sample_idxs(dataset.idx_to_class,
                10, 300, 12, rand_N=True, rand_K=True) for _ in range(100)]
            torch.save(bench, self.clustering_benchmark)

# models used for real-world datasets
class RealAnchoredFilteringModel(AnchoredFilteringModel):
    def __init__(self, args):
        super().__init__(args)
        self.train_metrics = ['bcent']
        self.test_metrics = ['bcent']

    def get_dataset(self, train=True):
        raise NotImplementedError

    def get_classes(self, train=True):
        return None

    def build_train_loader(self):
        self.train_loader = cds.get_train_loader(self.get_dataset(),
                self.B, self.N, self.K, self.num_steps,
                classes=self.get_classes())

    def build_test_loader(self, filename=None):
        self.test_loader = cds.get_test_loader(self.get_dataset(False),
                (self.filtering_benchmark if filename is None else filename),
                classes=self.get_classes(False))

    def gen_benchmarks(self, force=False):
        if not os.path.isfile(self.filtering_benchmark) or force:
            print('generating benchmark {}...'.format(self.filtering_benchmark))

            dataset = cds.ClusteredDataset(self.get_dataset(False),
                    classes=self.get_classes(False))
            bench = [cds.sample_idxs(dataset.idx_to_class,
                10, 100, 4, rand_N=True, rand_K=True) for _ in range(100)]
            torch.save(bench, self.filtering_benchmark)

            print('generating benchmark {}...'.format(self.clustering_benchmark))
            bench = [cds.sample_idxs(dataset.idx_to_class,
                10, 300, 12, rand_N=True, rand_K=True) for _ in range(100)]
            torch.save(bench, self.clustering_benchmark)

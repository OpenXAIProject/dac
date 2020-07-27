import torch
import time
import logging
import numpy as np

def get_logger(header, filename):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(header)
    logger.addHandler(logging.FileHandler(filename, mode='w'))
    return logger

class Accumulator():
    def __init__(self, *args):
        self.args = args
        self.argnum = {}
        for i, arg in enumerate(args):
            self.argnum[arg] = i
        self.sums = [0]*len(args)
        self.cnts = [0]*len(args)
        self.clock = time.time()

    def update(self, arg, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.sums[self.argnum[arg]] += val
        self.cnts[self.argnum[arg]] += 1

    def update_all(self, vals):
        try:
            iter(vals)
        except:
            vals = [vals]
        else:
            vals = vals
        vals = [v for v in vals if v is not None]
        assert(len(vals) == len(self.sums))

        for i, v in enumerate(vals):
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.sums[i] += v
            self.cnts[i] += 1

    def reset(self):
        self.sums = [0]*len(self.args)
        self.cnts = [0]*len(self.args)
        self.clock = time.time()

    def get(self, arg):
        i = self.argnum.get(arg)
        if i is not None:
            if self.cnts[i] == 0:
                return 0
            else:
                return self.sums[i]/self.cnts[i]
        else:
            return None

    def info(self, header=None, epoch=None, it=None, show_et=True):
        et = time.time() - self.clock
        line = '' if header is None else header + ': '
        if epoch is not None:
            line += 'epoch {:d}, '.format(epoch)
        if it is not None:
            line += 'iter {:d}, '.format(it)
        for arg in self.args:
            val = self.get(arg)
            if type(val) == float or type(val) == np.float32 or type(val) == np.float64:
                line += '{} {:.4f}, '.format(arg, val)
            else:
                line += '{} {}, '.format(arg, val)
        if show_et:
            line += '({:.3f} secs)'.format(et)

        return line

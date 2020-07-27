import imp
import os
import numpy as np

def add_args(args1, args2):
    for k, v in args2.__dict__.items():
        args1.__dict__[k] = v

def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    module = imp.load_source(module_name, filename)
    return module, module_name

# adj: N * N
def adj_to_labels(adj):
    N, N = adj.shape
    labels = np.zeros(N, dtype=np.long)
    k = 0
    processed = np.zeros(N, dtype=np.bool)
    for i in range(N):
        if not processed[i]:
            processed[i] = True
            labels[i] = k
            idxs = adj[i] > 0.5
            labels[idxs] = k
            processed[idxs] = True
            k = k + 1
    return labels

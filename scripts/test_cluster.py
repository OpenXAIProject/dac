import torch
import os
import time
import argparse
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import numpy as np

from utils.log import get_logger, Accumulator
from utils.misc import load_module
from utils.paths import results_path
from utils.tensor import to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='models/mog.py')
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--filename', type=str, default='test_cluster.log')
parser.add_argument('--gpu', type=str, default='0')
args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

module, module_name = load_module(args.model)
model = module.load(args)
print(str(args))

save_dir = os.path.join(results_path, module_name, args.run_name)
model.cuda()
model.load(os.path.join(save_dir, 'model.tar'))

model.build_test_loader(filename=model.clustering_benchmark)
accm = Accumulator('ll', 'oracle', 'ARI', 'NMI', 'k-MAE', 'et')
num_failure = 0
logger = get_logger('{}_{}'.format(module_name, args.run_name),
        os.path.join(save_dir, args.filename))

import matplotlib.pyplot as plt
from utils.plots import scatter, scatter_mog, draw_ellipse

for batch in tqdm(model.test_loader):
    tick = time.time()
    results = model.cluster(batch['X'].cuda(),
            max_iter=args.max_iter, verbose=False)
    et = (time.time() - tick) / len(batch['X'])

    true_labels = to_numpy(batch['oh_labels'].argmax(-1))
    ari = 0
    nmi = 0
    mae = 0
    labels = results['labels']
    for b in range(len(labels)):
        labels_b = to_numpy(labels[b])
        ari += ARI(true_labels[b], labels_b)
        nmi += NMI(true_labels[b], labels_b, average_method='arithmetic')
        mae += abs(len(np.unique(true_labels[b])) - len(np.unique(labels_b)))
    ari /= len(labels)
    nmi /= len(labels)
    mae /= len(labels)

    accm.update('ARI', ari)
    accm.update('NMI', nmi)
    accm.update('k-MAE', mae)

    ll = results.get('ll', 0.0)
    oracle = batch.get('ll', 0.0)

    accm.update_all([ll, oracle, ari, nmi, mae, et])

logger.info(accm.info())

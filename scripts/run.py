import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import argparse
import json

from utils.log import get_logger, Accumulator
from utils.misc import load_module
from utils.paths import results_path, benchmarks_path

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='models/mog.py')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--clip', type=float, default=10.0)
parser.add_argument('--save_all', action='store_true')
parser.add_argument('--regen_benchmarks', action='store_true')
parser.add_argument('--mode', type=str, choices=['train', 'test', 'vis'], default='train')
args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

module, module_name = load_module(args.model)
model = module.load(args)
exp_id = '{}:{}'.format(module_name, args.run_name)
save_dir = os.path.join(results_path, module_name, args.run_name)

if not os.path.isdir(benchmarks_path):
    os.makedirs(benchmarks_path)
model.gen_benchmarks(force=args.regen_benchmarks)
model.cuda()

def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save hyperparams
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)

    model.build_train_loader()
    model.build_test_loader()
    model.build_optimizer()

    train_accm = Accumulator(*model.train_metrics)
    test_accm = Accumulator(*model.test_metrics)
    logger = get_logger(exp_id, os.path.join(save_dir,
        'train_'+time.strftime('%Y%m%d-%H%M')+'.log'))

    for t, batch in enumerate(model.train_loader, 1):
        model.train_batch(batch, train_accm)

        if t % args.test_freq == 0:
            line = 'step {}, '.format(t)
            line += model.get_lr_string() + ', '
            line += train_accm.info(header='train', show_et=False)
            model.test(test_accm)
            line += test_accm.info(header='test', )
            logger.info(line)
            train_accm.reset()
            test_accm.reset()

        if t % args.save_freq == 0:
            model.save(os.path.join(save_dir, 'model.tar'))

    model.save(os.path.join(save_dir, 'model.tar'))

def test():
    model.load(os.path.join(save_dir, 'model.tar'))
    model.build_test_loader()
    accm = Accumulator(*model.test_metrics)
    logger = get_logger(exp_id, os.path.join(save_dir, 'test.log'))
    model.test(accm)
    logger.info(accm.info(header='test'))

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        vis()

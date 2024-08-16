'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''

import os
from os.path import join
from enum import Enum
from parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

config = {}
config['batch_size'] = args.batch_size
config['lr'] = args.lr
config['dataset'] = args.dataset
config['num_layers'] = args.layer
config['emb_dim'] = args.emb_dim
config['model'] = args.model
config['option'] = args.option
config['decay'] = args.decay
config['epochs'] = args.epochs
config['top_k'] = args.top_k
config['verbose'] = args.verbose
config['epochs_per_eval'] = args.epochs_per_eval
config['epochs_per_lr_decay'] = args.epochs_per_lr_decay
config['seed'] = args.seed
config['win'] = args.win
config['r_beta'] = args.r_beta
config['a_beta'] = args.a_beta
config['a_method'] = args.a_method
config['r_method'] = args.r_method
config['by_time'] = args.by_time
config['loadedModel'] = args.loadedModel
config['drop'] = args.drop
config['test_size'] = args.test_size
config['num_exp'] = args.num_exp
config['min_u_ratings'] = args.min_u_ratings
config['save'] = args.save

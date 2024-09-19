'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog="tempLGCN", description="Dynamic GCN-based CF recommender")
    parser.add_argument('--model', type=str, default='tempLGCN', help='rec-model, support [tempLGCN, LightGCN, MCCF]')
    parser.add_argument('--option', type=str, default='lgcn_b_ar', help='rec-model, support [lgcn, lgcn_b, lgcn_b_a, lgcn_b_r, lgcn_b_ar]')
    parser.add_argument('--dataset', type=str, default='ml-100k', help="available datasets: [ml100k, ml1m, ml10m, yelp, douban, amazon]")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--emb_dim', type=int, default=64, help="the embedding size for learning parameters")
    parser.add_argument('--layer', type=int, default=0, help="the layer num of GCN")
    parser.add_argument('--batch_size', type=int, default= 9000, help="the batch size for bpr loss training procedure")
    parser.add_argument('--epochs', type=int,default=101)
    parser.add_argument('--epochs_per_eval', type=int,default=25)
    parser.add_argument('--epochs_per_lr_decay', type=int,default=10)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-06, help="the weight decay for l2 normalizaton")
    parser.add_argument('--path', type=str, default="./checkpoints", help="path to save weights")
    parser.add_argument('--top_k', type=int, default=20, help="@k test list")
    parser.add_argument('--win', type=float, default=1)
    parser.add_argument('--r_method', type=str, default='exp')
    parser.add_argument('--r_beta', type=float, default=25)
    parser.add_argument('--a_beta', type=float, default=0.055)
    parser.add_argument('--a_method', type=str, default='log')
    parser.add_argument('--by_time', type=bool, default=False)
    parser.add_argument('--loadedModel', type=bool, default=False)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--min_u_ratings', type=int, default=20)
    parser.add_argument('--save', type=bool, default=False)
    
    return parser.parse_args()

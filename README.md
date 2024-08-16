
## LGCNplus-pytorch (we named the model as tempLGCN in the paper but the repository name is LGCNplus)
06 Feb 2024
This is the Pytorch implementation for "tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks".

## Introduction

User’s taste evolves, and failing to account for this can lead to less effective recommendations. To tackle this, a novel temporal collaborative filtering model based on a graph convolutional network (GCN) is introduced. This model incorporates long-term and transient temporal signals via relative and absolute time functions, designed to capture the changes in user preferences. Experiments on MovieLens datasets showcase the superiority of this approach over state-of-the-art methods.

## Enviroment Requirement

`pip install -r requirements.txt`

## Dataset

We tested on three datasets: MovieLens ml100k and ml1m.

## An example to run a 3-layer tempLGCN

run tempLGCN on **ML100k** dataset (ml100k is the default dataset):

python main.py

*NOTE*, here are some key parameters:

1. --model sets the model to run. The options are: 'lgcn' - LightGCN, 'lgcn_b' - LGCN with baseline signal, 'lgcn_b_a' - LGCN with baseline and absolute temporal signals, 'lgcn_b_r' - LGCN with baseline and relative temporal signals, 'lgcn_b_ar' - tempLGCN i.e., the full model including the baseline and absolute and relative temporal signals. The default value: 'lgcn_b_ar'

2. --dataset chooses the dataset to run. The options are: 'ml100k' and 'ml1m'. The default value: 'ml100k'.

3. --batch_size sets the batch size. We chose 95000 for ml100k and 950000 for ml1m making a fullbatch. The default value: 95000.

4. --epochs sets the number of epochs to run. The default value: 991.

5. --layers sets the number of layers for GCN. The default value: 4

6. --decay sets the weight decay. The default value: 1e-05 (please set it to 1e-06 for ml1m)

7. --a_method sets the function for absolute temporal function. The options are: 'exp' and 'log'. The default value: 'log'

8. --r_method sets the function for relative temporal function. The options are: 'exp' and 'log'. The default value: 'exp'

9. --a_beta sets the beta parameter for absolute temporal function. The default value: 0.055

10. --r_beta sets the beta parameter for relative temporal function. The default value: 0.025

11. --epochs_per_eval sets the number of epochs that evalution to be taken per. The default value: 10

12. --emb_dim sets the dimension size of embeddings. The default value: 64.

13. --top_k sets the number of top k value. The default value: 20

# Run for the best results:

# for ml-100k
python main.py  --batch_size=90000 --epochs=1201  --epochs_per_eval=25  --layer=4  --decay=1e-05  --model=lgcn_b_ar --dataset=ml-100k  --a_method=log --a_beta=0.05 --r_method=exp --r_beta=0.35

# for ml-1m
python main.py --batch_size=950000  --model=lgcn_b_ar --dataset=ml-1m --epochs=851 --layer=5  --decay=1e-06  --a_beta=0.06 --r_method='exp' --epochs_per_eval=25 --r_beta=5 --emb_dim=500

# for douban-book
main.py  --batch_size=400000 --epochs=201  --epochs_per_eval=5  --layer=0  --decay=1e-03  --model=lgcn  --a_beta=0.001 --a_method=exp --r_beta=0.008 --dataset=douban_book --top_k=5

## Extend:
* If you want to run tempLGCN on your own dataset, you can just feed any data that has "user item rating timestamp" format and use --dataset parameter to provide the name of your dataset. Datasets are stored in the data subfolder.

## Results of ML-100k:

  RMSE:          0.8877, 0.8861, 0.8870, 0.8865, 0.8869 -> 0.8868

  Recall@20:     0.6670, 0.6660, 0.6650, 0.6630, 0.6660 -> 0.6654

  Prec@20:       0.6810, 0.6800, 0.6760, 0.6810, 0.6830 -> 0.6802


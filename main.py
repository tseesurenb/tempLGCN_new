#%%
import numpy as np
import torch
import data_prep as dp
from world import config
from utils import train_model

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

SEED = config['seed']
# Set a random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set global variables
R_BETA = config['r_beta']
R_METHOD = config['r_method']
A_BETA = config['a_beta']
A_METHOD = config['a_method']
DATASET = config['dataset']
VERBOSE = config['verbose']
MODEL = config['model']
NUM_EXP = config['num_exp']
MIN_USER_RATINGS = config['min_u_ratings']
MODEL_OPTION = config['option']
NUM_LAYERS = config['num_layers']
DECAY = config['decay']

# STEP 1: loading dataset
if VERBOSE:
    print(f'loading {DATASET} ...')

df, _, _, stats = dp.load_data(dataset=DATASET, min_interaction_threshold=MIN_USER_RATINGS, verbose=VERBOSE)

NUM_USERS, NUM_ITEMS, MEAN_RATING, NUM_RATINGS, TIME_DISTANCE = stats['num_users'], stats['num_items'], stats['mean_rating'], stats['num_ratings'], stats['time_distance']

# STEP 2: adding absolute and relative decays for users and items
df = dp.add_abs_decay(df, method=A_METHOD, beta=A_BETA, verbose=VERBOSE)
df = dp.add_u_rel_decay(df, method=R_METHOD, beta=R_BETA, verbose=VERBOSE)
df = dp.add_i_rel_decay(df, method=R_METHOD, beta=R_BETA, verbose=VERBOSE)

# STEP 3: getting the interaction matrix values
rmat_data = dp.get_rmat_values(df, verbose=VERBOSE)
 
seeds = [7, 12, 89, 91, 41]

rmses, recalls, precs, ncdgs_5, ncdgs_10, ncdgs_15, ncdgs_20 = [], [], [], [], [], [], []

exp_n = 1

for seed in seeds:
    print(f'Experiment ({exp_n}) starts with seed:{seed}')
    rmse, recall, prec, ncdg = train_model(model=MODEL, data=rmat_data, data_stats=stats, config=config, t_seed=seed)

    rmses.append(float(rmse))
    recalls.append(float(recall))
    precs.append(float(prec))
    ncdgs_5.append(float(ncdg["@5"]))
    ncdgs_10.append(float(ncdg["@10"]))
    ncdgs_15.append(float(ncdg["@15"]))
    ncdgs_20.append(float(ncdg["@20"]))

    exp_n += 1

print(f"\nModel: {br}{MODEL}{rs} | Option: {br}{MODEL_OPTION}{rs} | Dataset: {br}{DATASET}{rs} | Layers: {br}{NUM_LAYERS}{rs} | Decay: {br}{DECAY}{rs} | Batch: {br}{config['batch_size']}{rs}")
print(f" Temp: {br}{A_METHOD}{rs} - {br}{A_BETA}{rs} | {br}{R_METHOD}{rs} - {br}{R_BETA}{rs}")


print(f"   RMSE: {rmses[0]:.4f}, {rmses[1]:.4f}, {rmses[2]:.4f}, {rmses[3]:.4f}, {rmses[4]:.4f} | {round(np.mean(rmses), 4):.4f}, {round(np.std(rmses), 4):.4f}")
print(f" Recall: {recalls[0]:.4f}, {recalls[1]:.4f}, {recalls[2]:.4f}, {recalls[3]:.4f}, {recalls[4]:.4f} | {round(np.mean(recalls), 4):.4f}, {round(np.std(recalls), 4):.4f}")
print(f"   Prec: {precs[0]:.4f}, {precs[1]:.4f}, {precs[2]:.4f}, {precs[3]:.4f}, {precs[4]:.4f} | {round(np.mean(precs), 4):.4f}, {round(np.std(precs), 4):.4f}")
print(f" NDCG@5: {ncdgs_5[0]:.4f}, {ncdgs_5[1]:.4f}, {ncdgs_5[2]:.4f}, {ncdgs_5[3]:.4f}, {ncdgs_5[4]:.4f} | {round(np.mean(ncdgs_5), 4):.4f}, {round(np.std(ncdgs_5), 4):.4f}")
print(f"NDCG@10: {ncdgs_10[0]:.4f}, {ncdgs_10[1]:.4f}, {ncdgs_10[2]:.4f}, {ncdgs_10[3]:.4f}, {ncdgs_10[4]:.4f} | {round(np.mean(ncdgs_10), 4):.4f}, {round(np.std(ncdgs_10), 4):.4f}")
print(f"NDCG@15: {ncdgs_15[0]:.4f}, {ncdgs_15[1]:.4f}, {ncdgs_15[2]:.4f}, {ncdgs_15[3]:.4f}, {ncdgs_15[4]:.4f} | {round(np.mean(ncdgs_15), 4):.4f}, {round(np.std(ncdgs_15), 4):.4f}")
print(f"NDCG@20: {ncdgs_20[0]:.4f}, {ncdgs_20[1]:.4f}, {ncdgs_20[2]:.4f}, {ncdgs_20[3]:.4f}, {ncdgs_20[4]:.4f} | {round(np.mean(ncdgs_20), 4):.4f}, {round(np.std(ncdgs_20), 4):.4f}")

#%%
import numpy as np
import torch
from torch import nn, optim
import time
from tqdm import tqdm

import data_prep as dp
from world import config
from model import LGCN_full, LGCN
from utils import get_recall_at_k, minibatch, save_model, calculate_ndcg, plot_loss

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
LR = config['lr']
EPOCHS = config['epochs']
EPOCHS_PER_EVAL= config['epochs_per_eval']
DECAY = config['decay']
NUM_LAYERS = config['num_layers']
EMB_DIM = config['emb_dim']
BATCH_SIZE = config['batch_size']
TOP_K = config['top_k']
R_BETA = config['r_beta']
R_METHOD = config['r_method']
A_BETA = config['a_beta']
A_METHOD = config['a_method']
DATASET = config['dataset']
VERBOSE = config['verbose']
MODEL = config['model']
TEST_SIZE = config['test_size']
NUM_EXP = config['num_exp']
MIN_USER_RATINGS = config['min_u_ratings']
SAVE_MODEL = config['save']

if VERBOSE:
    print(f'loading {DATASET} ...')

# STEP 1: loading dataset
df, _, _, stats = dp.load_data(dataset=DATASET, min_interaction_threshold=MIN_USER_RATINGS, verbose=VERBOSE)
NUM_USERS, NUM_ITEMS, MEAN_RATING, NUM_RATINGS, TIME_DISTANCE = stats['num_users'], stats['num_items'], stats['mean_rating'], stats['num_ratings'], stats['time_distance']

# STEP 2: adding absolute and relative decays for users and items
df = dp.add_abs_decay(df, method=A_METHOD, beta=A_BETA, verbose=VERBOSE)
df = dp.add_u_rel_decay(df, method=R_METHOD, beta=R_BETA, verbose=VERBOSE)
df = dp.add_i_rel_decay(df, method=R_METHOD, beta=R_BETA, verbose=VERBOSE)

# STEP 3: getting the interaction matrix values
rmat_data = dp.get_rmat_values(df, verbose=VERBOSE)

# STEP 4: splitting the data into train and test sets
rmat_train_data, rmat_val_data = dp.train_test_split_by_user(rmat_data, test_size=TEST_SIZE, seed=SEED, verbose=VERBOSE)

# STEP 5: convert the interaction matrix to adjacency matrix
edge_train_data = dp.rmat_2_adjmat_faster(NUM_USERS, NUM_ITEMS, rmat_train_data)
edge_val_data = dp.rmat_2_adjmat_faster(NUM_USERS, NUM_ITEMS, rmat_val_data)

# STEP 6: get the interaction matrix values
r_mat_train_idx = rmat_train_data['rmat_index']
r_mat_train_v = rmat_train_data['rmat_values']
r_mat_train_rts = rmat_train_data['rmat_ts']
r_mat_train_abs_decay = rmat_train_data['rmat_abs_decay']
r_mat_train_u_rel_decay = rmat_train_data['rmat_u_rel_decay']
r_mat_train_i_rel_decay = rmat_train_data['rmat_i_rel_decay']

r_mat_val_idx = rmat_val_data['rmat_index']
r_mat_val_v = rmat_val_data['rmat_values']
r_mat_val_rts = rmat_val_data['rmat_ts']
r_mat_val_abs_decay = rmat_val_data['rmat_abs_decay']
r_mat_val_u_rel_decay = rmat_val_data['rmat_u_rel_decay']
r_mat_val_i_rel_decay = rmat_val_data['rmat_i_rel_decay']

# STEP 7: setting the loss variables
train_losses = []
val_losses = []

# STEP 8: setting the evaluation variables
val_recall = []
val_prec = []
val_ncdg_5 = []
val_ncdg_10 = []
val_ncdg_15 = []
val_ncdg_20 = []
val_rmse = []
train_rmse = []

 # STEP 9: setting the message passing index
train_edge_index = edge_train_data['edge_index']
val_edge_index = edge_val_data['edge_index']

# STEP 10: setting the supervision data
train_src = r_mat_train_idx[0]
train_dest = r_mat_train_idx[1]
train_values = r_mat_train_v
train_abs_decay = r_mat_train_abs_decay
train_u_rel_decay = r_mat_train_u_rel_decay
train_i_rel_decay = r_mat_train_i_rel_decay
val_src = r_mat_val_idx[0]
val_dest = r_mat_val_idx[1]
val_values = r_mat_val_v
val_abs_decay = r_mat_val_abs_decay
val_u_rel_decay = r_mat_val_u_rel_decay
val_i_rel_decay = r_mat_val_i_rel_decay

# STEP 11: setting the model
model = LGCN(num_users=NUM_USERS,
                num_items=NUM_ITEMS,
                num_layers=NUM_LAYERS,
                embedding_dim = EMB_DIM,
                add_self_loops = False,
                mu = MEAN_RATING,
                model=MODEL,
                verbose=VERBOSE)


# STEP 12: setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if VERBOSE == 1:
    print(f"Device is - {device}")

model = model.to(device)

train_src = train_src.to(device)
train_dest = train_dest.to(device)
train_values = train_values.to(device)
train_abs_t_decay = train_abs_decay.to(device)
train_u_rel_decay = train_u_rel_decay.to(device)
train_i_rel_decay = train_i_rel_decay.to(device)

val_src = val_src.to(device)
val_dest = val_dest.to(device)
val_values = val_values.to(device)
val_abs_decay = val_abs_decay.to(device)
val_u_rel_decay = val_u_rel_decay.to(device)
val_i_rel_decay = val_i_rel_decay.to(device)

train_edge_index = train_edge_index.to(device)
val_edge_index = val_edge_index.to(device)

# STEP 13: setting the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr = LR, weight_decay=DECAY)
loss_func = nn.MSELoss()

# STEP 14: initialized minimum variables
min_RMSE = 1000
min_RMSE_epoch = 0
min_RECALL = 0
min_PRECISION = 0

all_compute_time = 0
avg_compute_time = 0
val_epochs = 0

# STEP 15: training the model
for epoch in tqdm(range(EPOCHS), position=1, mininterval=1.0, ncols=100):
        start_time = time.time()
        
        if len(train_src) != BATCH_SIZE:
            total_iterations = len(train_src) // BATCH_SIZE + 1
        else:
            total_iterations = len(train_src) // BATCH_SIZE
        
        model.train()
        train_loss = 0.
        
        # Generate batches of data using the minibatch function
        train_minibatches = minibatch(train_abs_decay, train_u_rel_decay, train_i_rel_decay, train_src, train_dest, train_values, batch_size=BATCH_SIZE)
        
        # Iterate over each batch using enumerate
        for b_abs_decay, b_u_rel_decay, b_i_rel_decay, b_src, b_dest, b_values in train_minibatches:
            b_pred_ratings = model.forward(train_edge_index, b_src, b_dest, b_abs_decay, b_u_rel_decay, b_i_rel_decay)
            b_loss = loss_func(b_pred_ratings, b_values)
            train_loss += b_loss
            
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
        
        train_loss = train_loss / total_iterations
        
        if epoch %  EPOCHS_PER_EVAL == 0:
            
            train_rmse.append(np.sqrt(train_loss.item()))  
            model.eval()
            
            val_epochs += 1
            
            with torch.no_grad():      
                val_loss = 0.   
                val_pred_ratings = []
                
                if len(val_src) != BATCH_SIZE:
                    total_iterations = len(val_src) // BATCH_SIZE + 1
                else:
                    total_iterations = len(val_src) // BATCH_SIZE

                
                val_mini_batches = minibatch(val_abs_decay, val_u_rel_decay, val_i_rel_decay, val_src, val_dest, val_values, batch_size=BATCH_SIZE)
                
                for b_abs_decay, b_u_rel_decay, b_i_rel_decay, b_src, b_dest, b_values in val_mini_batches:
                        
                    b_pred_ratings = model.forward(val_edge_index, b_src, b_dest, b_abs_decay, b_u_rel_decay, b_i_rel_decay)
                    
                    val_b_loss = loss_func(b_pred_ratings, b_values)
                    val_loss += val_b_loss
                    val_pred_ratings.extend(b_pred_ratings)
        
                val_loss = val_loss / total_iterations
                
                recall, prec = get_recall_at_k(r_mat_val_idx,
                                                            r_mat_val_v,
                                                            torch.tensor(val_pred_ratings),
                                                            k=TOP_K)
                
                ncdg_5 = calculate_ndcg(r_mat_val_idx, r_mat_val_v, val_pred_ratings, k=5)
                ncdg_10 = calculate_ndcg(r_mat_val_idx, r_mat_val_v, val_pred_ratings, k=10)
                ncdg_15 = calculate_ndcg(r_mat_val_idx, r_mat_val_v, val_pred_ratings, k=15)
                ncdg_20 = calculate_ndcg(r_mat_val_idx, r_mat_val_v, val_pred_ratings, k=20)
                
                recall = round(recall, 3)
                prec = round(prec, 3)
                val_recall.append(recall)
                val_prec.append(prec)
                val_ncdg_5.append(ncdg_5)
                val_ncdg_10.append(ncdg_10)
                val_ncdg_15.append(ncdg_15)
                val_ncdg_20.append(ncdg_20)
                val_rmse.append(np.sqrt(val_loss.item()))
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                
                f_train_loss = "{:.4f}".format(round(np.sqrt(train_loss.item()), 4))
                f_val_loss = "{:.4f}".format(round(np.sqrt(val_loss.item()), 4))
                f_recall = "{:.4f}".format(round(recall, 4))
                f_precision = "{:.4f}".format(round(prec, 4))
                f_ncdg_5 = "{:.4f}".format(round(ncdg_5, 4))
                f_ncdg_10 = "{:.4f}".format(round(ncdg_10, 4))
                f_ncdg_15 = "{:.4f}".format(round(ncdg_15, 4))
                f_ncdg_20 = "{:.4f}".format(round(ncdg_20, 4))
                
                if (recall + prec) != 0:
                    f_f1_score = "{:.4f}".format(round((2*recall*prec)/(recall + prec), 4))
                else:
                    f_f1_score = 0
                    
                f_time = "{:.2f}".format(round(time.time() - start_time, 2))
                f_epoch = "{:4.0f}".format(epoch)
                            
                if min_RMSE > np.sqrt(val_loss.item()):
                    if SAVE_MODEL:
                        save_model(model, 'models/' + DATASET, '_model.pt')
                    min_RMSE = np.sqrt(val_loss.item())
                    min_RMSE_loss = f_val_loss
                    min_RMSE_epoch = epoch
                    min_RECALL_f = f_recall
                    min_PRECISION_f = f_precision
                    min_RECALL = recall
                    min_PRECISION = prec
                    min_F1 = f_f1_score
                    min_ncdg_5 = round(ncdg_5, 4)
                    min_ncdg_10 = round(ncdg_10, 4)
                    min_ncdg_15 = round(ncdg_15, 4)
                    min_ncdg_20 = round(ncdg_20, 4)
                    min_ncdg = {"@5": min_ncdg_5, "@10": min_ncdg_10, "@15": min_ncdg_15, "@20": min_ncdg_20}

                trace = True
                if epoch %  (EPOCHS_PER_EVAL) == 0 and trace == True:
                    tqdm.write(f"[Epoch {f_epoch} - {f_time}, {avg_compute_time}]\tRMSE(train -> val): {f_train_loss}"
                            f" -> \033[1m{f_val_loss}\033[0m | "
                            f"Recall, Prec:{f_recall, f_precision}, NCDG: @5 {f_ncdg_5} | @10 {f_ncdg_10} | @15 {f_ncdg_15} | @20 {f_ncdg_20}")
                
        all_compute_time += (time.time() - start_time)
        avg_compute_time = "{:.4f}".format(round(all_compute_time/(epoch+1), 4)) 


tqdm.write(f"\nModel: {br}{MODEL}{rs} | DATASET: {br}{DATASET}{rs} | Layers: {br}{NUM_LAYERS}{rs} | DECAY: {br}{DECAY}{rs}")
tqdm.write(f"Temp: {br}{A_METHOD}{rs} - {br}{A_BETA}{rs} | {br}{R_METHOD}{rs} - {br}{R_BETA}{rs}")
tqdm.write(f"RMSE: {br}{min_RMSE_loss} at epoch {min_RMSE_epoch}{rs} with Recall, Precision: {br}{min_RECALL_f, min_PRECISION_f}{rs} | NCDG: {br}{min_ncdg}{rs}")
#plot_loss(val_epochs, train_losses, val_losses, train_rmse, val_rmse, val_recall, val_prec)
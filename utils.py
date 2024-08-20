'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''

import torch
from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
import numpy as np
import sys

from torch import nn, optim
import time
from tqdm import tqdm
import data_prep as dp
from model import LGCN_full, tempLGCN, MCCF, tempLGCN2

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

models = {
    'tempLGCN': tempLGCN,
    'tempLGCN2': tempLGCN2,
    'MCCF': MCCF
}

def get_recall_at_k(input_edge_index,
                    input_edge_values,
                    pred_ratings,
                    k=10,
                    threshold=3.5):
    user_item_rating_list = defaultdict(list)
    
    for i in range(len(input_edge_index[0])):
        src = input_edge_index[0][i].item()
        dest = input_edge_index[1][i].item()
        true_rating = input_edge_values[i].item()
        pred_rating = pred_ratings[i].item()
        
        user_item_rating_list[src].append((pred_rating, true_rating))
        
    recalls = dict()
    precisions = dict()

    for user_id, user_ratings in user_item_rating_list.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((pred_r >= threshold) for (pred_r, _) in user_ratings[:k])
        
        n_rel_and_rec_k = sum(((true_r >= threshold) and (pred_r >= threshold)) \
                                for (pred_r, true_r) in user_ratings[:k])
        
        precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
    overall_recall = sum(rec for rec in recalls.values()) / len(recalls)
    overall_precision = sum(prec for prec in precisions.values()) / len(precisions)
    
    #print("Calculating recall and precision...")
    #print(f"Threshold: {threshold}", "Top-K: ", k, "len of recalls: ", len(recalls), "len of precisions: ", len(precisions))
    #print("Overall Recall: ", overall_recall, "Overall Precision: ", overall_precision)
    
    return overall_recall, overall_precision

def get_top_k(input_edge_index,
                    input_edge_values,
                    pred_ratings,
                    k=10,
                    threshold=3.5):
    user_item_rating_list = defaultdict(list)
    
    for i in range(len(input_edge_index[0])):
        src = input_edge_index[0][i].item()
        dest = input_edge_index[1][i].item()
        true_rating = input_edge_values[i].item()
        pred_rating = pred_ratings[i].item()
        
        user_item_rating_list[src].append((pred_rating, true_rating))
        
    recalls = dict()
    precisions = dict()

    for user_id, user_ratings in user_item_rating_list.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((pred_r >= threshold) for (pred_r, _) in user_ratings[:k])
        
        n_rel_and_rec_k = sum(((true_r >= threshold) and (pred_r >= threshold)) \
                                for (pred_r, true_r) in user_ratings[:k])
        
        precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
    overall_recall = sum(rec for rec in recalls.values()) / len(recalls)
    overall_precision = sum(prec for prec in precisions.values()) / len(precisions)
    
    #print("Calculating recall and precision...")
    #print(f"Threshold: {threshold}", "Top-K: ", k, "len of recalls: ", len(recalls), "len of precisions: ", len(precisions))
    #print("Overall Recall: ", overall_recall, "Overall Precision: ", overall_precision)
    
    return overall_recall, overall_precision

def print_rmse(ITERATIONS, iter, train_loss, val_loss, recall, precision, time):
    
    f_train_loss = "{:.3f}".format(round(np.sqrt(train_loss.item()), 3))
    f_val_loss = "{:.3f}".format(round(np.sqrt(val_loss.item()), 3))
    f_recall = "{:.3f}".format(round(recall, 3))
    f_precision = "{:.3f}".format(round(precision, 3))
    f_time = "{:.2f}".format(round(time, 2))
    f_iter = "{:.0f}".format(iter)
    
    sys.stdout.write(f"\rEpoch {f_iter}/{ITERATIONS} - Train Loss: {train_loss:.3f}, "
                     f"Val Loss: {val_loss:.3f}, Recall: {f_recall}, Precision: {f_precision}, Time: {f_time} s")
    sys.stdout.flush()
    
    #print(f"[Epoch ({f_time}) {f_iter}]\tRMSE(train->val): {f_train_loss}"
    #      f"\t-> {f_val_loss} | "
    #      f"Recall, Prec:{f_recall, f_precision}")
  

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 1024)
    
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Save a PyTorch model to a target directory.

    Args:
        model (torch.nn.Module): _description_
        target_dir (str): _description_
        model_name (str): _description_
    """
    
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    
    # Save the model state_dict()
    # print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    
    # Save the embeddings
    embeddings_save_path = target_dir_path / "_embeddings.pt"
    model.save_embeddings(embeddings_save_path)

def load_model(model, model_path, model_name="_model.pt"):
    """Load a PyTorch model from a file.
    
    Args:
        model_class (torch.nn.Module): The class of the model to be loaded.
        model_path (str): The path to the saved model file.
    
    Returns:
        model (torch.nn.Module): The loaded model.
    """
    # Load the saved state dictionary
    print(f'path to the model: {model_path}')
    state_dict = torch.load(model_path + model_name)
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Load the embeddings
    embeddings_save_path = model_path + "_embeddings.pt"
    model.load_embeddings(embeddings_save_path)
    
    return model

def predict(model: torch.nn.Module,
            user_id: int,
            top_k: int,
            device: torch.device):
    
    model.to(device)
    
    model.eval()
    with torch.inference_mode():
        # make prediction
        top_k_pred = model(user_id, top_k)

    return top_k_pred

def predict_for_user(model, test_data, ratings_df, items_df, pred_user_id, device):
    
    unique_user_id = ratings_df['userId'].unique()
    unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id, 
    'mappedUserId': pd.RangeIndex(len(unique_user_id))
    })
    
    unique_item_id = ratings_df['itemId'].unique()
    unique_item_id = pd.DataFrame(data={
    'itemId': unique_item_id,
    'mappedItemId': pd.RangeIndex(len(unique_item_id))
    })
    
    # Your mappedUserId
    mapped_user_id = unique_user_id[unique_user_id['userId'] == pred_user_id]['mappedUserId'].values[0]

    # Select items that you haven't seen before
    items_rated = ratings_df[ratings_df['mappedUserId'] == mapped_user_id]
    items_not_rated = items_df[~items_df.index.isin(items_rated['itemId'])]
    items_not_rated = items_not_rated.merge(unique_item_id, on='itemId')
    item = items_not_rated.sample(1)

    print(f"The item we want to predict a raiting for is:  {item['title'].item()}")
    
    edge_label_index = torch.tensor([
    mapped_user_id,
    item.mappedItemId.item()])

    with torch.no_grad():
        test_data.to(device)
        pred = model(test_data.x_dict, test_data.edge_index_dict, edge_label_index)
        pred = pred.clamp(min=0, max=5).detach().cpu().numpy()
        
    return item, mapped_user_id, pred.item(), edge_label_index

def calculate_dcg_at_k(ratings, k):
    dcg = 0.0
    for i in range(min(k, len(ratings))):
        rel = ratings[i]
        dcg += (2 ** rel - 1) / np.log2(i + 2)
        #dcg += (rel) / np.log2(i + 1)
        
    return dcg

def calculate_ndcg(input_edge_index, input_edge_values, pred_ratings, k=20):
    user_item_rating_list = defaultdict(list)
    
    for i in range(len(input_edge_index[0])):
        src = input_edge_index[0][i].item()
        true_rating = input_edge_values[i].item()
        pred_rating = pred_ratings[i].item()
        
        user_item_rating_list[src].append((pred_rating, true_rating))
        
    ndcgs = []
    
    for user_id, user_ratings in user_item_rating_list.items():
        # Sort user ratings by predicted rating in descending order
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Extract true ratings from sorted user ratings
        true_ratings = [true_r for _, true_r in user_ratings]
        
        # Calculate DCG at k
        dcg = calculate_dcg_at_k(true_ratings, k)
        
        # Sort true ratings in descending order (ideal ranking)
        true_ratings.sort(reverse=True)
        
        # Calculate ideal DCG at k
        idcg = calculate_dcg_at_k(true_ratings, k)
        
        # Calculate NDCG
        if idcg == 0:
            ndcg = 0.0  # If IDCG is zero, set NDCG to zero
        else:
            ndcg = dcg / idcg
        
        ndcgs.append(ndcg)
    
    # Compute average NDCG across all users
    average_ndcg = np.mean(ndcgs)
    
    return average_ndcg

def plot_loss(epochs, train_loss, val_loss, train_rmse, val_rmse, recall, precision):
    epoch_list = [(i+1) for i in range(epochs)]
        
    # Plot for losses
    plt.figure(figsize=(21, 5))  # Adjust figure size as needed
    
    # Subplot for losses
    plt.subplot(1, 3, 1)
    plt.plot(epoch_list, train_loss, label='Total Training Loss')
    plt.plot(epoch_list, val_loss, label='Total Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Losses')
    plt.legend()

    # Subplot for RMSE
    plt.subplot(1, 3, 2)
    plt.plot(epoch_list, train_rmse, label='Training RMSE')
    plt.plot(epoch_list, val_rmse, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training vs Validation RMSE')
    plt.legend()

    # Subplot for metrics
    plt.subplot(1, 3, 3)
    plt.plot(epoch_list, recall, label='Recall')
    plt.plot(epoch_list, precision, label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Metrics')
    plt.legend()

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


def train_model(model, data, data_stats, config, t_seed):
    
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
    MODEL_OPTION = config['option']
    TEST_SIZE = config['test_size']
    NUM_EXP = config['num_exp']
    MIN_USER_RATINGS = config['min_u_ratings']
    SAVE_MODEL = config['save']
    NUM_USERS, NUM_ITEMS, MEAN_RATING, NUM_RATINGS, TIME_DISTANCE = data_stats['num_users'], data_stats['num_items'], data_stats['mean_rating'], data_stats['num_ratings'], data_stats['time_distance']
    
    # STEP 4: splitting the data into train and test sets
    rmat_train_data, rmat_val_data = dp.train_test_split_by_user(data, test_size=TEST_SIZE, seed=t_seed, verbose=VERBOSE)

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

    # STEP 12: setting the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if VERBOSE > -1:
            print(f"Device is - {device}")

    # STEP 11: setting the model

    model = models[MODEL](
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_layers=NUM_LAYERS,
        embedding_dim=EMB_DIM,
        add_self_loops=False,
        mu=MEAN_RATING,
        option=MODEL_OPTION,
        device=device,
        verbose=VERBOSE
    )
    
    model = model.to(device)

    train_src = train_src.to(device)
    train_dest = train_dest.to(device)
    train_values = train_values.to(device)
    train_abs_decay = train_abs_decay.to(device)
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
                    
                    recall = round(recall, 4)
                    prec = round(prec, 4)
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
                        min_RMSE = round(np.sqrt(val_loss.item()), 4)
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


    #tqdm.write(f"\nModel: {br}{MODEL}{rs} | Option: {br}{MODEL_OPTION}{rs} | Dataset: {br}{DATASET}{rs} | Layers: {br}{NUM_LAYERS}{rs} | Decay: {br}{DECAY}{rs}")
    #tqdm.write(f" Temp: {br}{A_METHOD}{rs} - {br}{A_BETA}{rs} | {br}{R_METHOD}{rs} - {br}{R_BETA}{rs}")
    tqdm.write(f" RMSE: {br}{min_RMSE_loss} at epoch {min_RMSE_epoch}{rs} with Recall@{TOP_K}, Prec@{TOP_K}: {br}{min_RECALL_f, min_PRECISION_f}{rs} | NCDG: {br}{min_ncdg}{rs}")
    
    return min_RMSE, min_RECALL, min_PRECISION, min_ncdg
    #plot_loss(val_epochs, train_losses, val_losses, train_rmse, val_rmse, val_recall, val_prec)
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

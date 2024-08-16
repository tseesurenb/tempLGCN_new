'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''

from torch import nn, Tensor
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_geometric.utils import degree


class LGCN(MessagePassing):    
    def __init__(self, 
                 model,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 num_layers=3,
                 add_self_loops = False,
                 mu = 0,
                 drop= 0,
                 device = 'cpu',
                 verbose = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.edge_index_norm = None
        self.verbose = verbose
        self.mu = mu
        self.model = model
        self.dropout = drop
        self.device = device
        
        print("Model:", model, " | Layers:", num_layers, " | emb dimension:", embedding_dim, " | dropout:", drop)
        
        self.user_baseline = False
        self.item_baseline = False
        self.u_abs_drift = False
        self.u_rel_drift = False
        
        self.users_emb_final = None
        self.items_emb_final = None
        
        if model == 'lgcn_b':  # lgcn + baseline
            self.item_baseline = True
            self.user_baseline = True
        elif model == 'lgcn_b_a':  # lgcn + baseline + absolute
            self.u_abs_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif model == 'lgcn_b_r':  # lgcn + baseline + relative
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif model == 'lgcn_b_ar': # lgcn + baseline + absolute + relative 
            self.u_abs_drift = True
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif model == 'lgcn_ar': # lgcn + baseline + absolute + relative 
            self.u_abs_drift = True
            self.u_rel_drift = True
        else: # pure lightGCN model only
            model = 'lgcn'
            self.mu = 0
        
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim).to(self.device)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim).to(self.device)
        
        self.users_emb.weight.requires_grad = True
        self.items_emb.weight.requires_grad = True
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        if self.user_baseline:
            self._u_base_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)
            nn.init.zeros_(self._u_base_emb.weight)
            self._u_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The user baseline embedding is ON.")
        
        if self.item_baseline:
            self._i_base_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim).to(self.device)
            nn.init.zeros_(self._i_base_emb.weight)
            self._i_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The item baseline embedding is ON.")

        if self.u_abs_drift:
            self._u_abs_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)  
            nn.init.zeros_(self._u_abs_drift_emb.weight)
            self._u_abs_drift_emb.weight.requires_grad = True
            if self.verbose:
                print("The absolute user drift temporal embedding is ON.")

        if self.u_rel_drift:
            self._u_rel_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)   
            nn.init.zeros_(self._u_rel_drift_emb.weight)
            self._u_rel_drift_emb.weight.requires_grad = True
            if self.verbose:
                print("The relative user drift temporal embedding is ON.")
                
        self.f = nn.ReLU()
        #self.f = nn.SiLU()
              
    def forward(self, edge_index: Tensor, src: Tensor, dest: Tensor, u_abs_decay: Tensor, u_rel_decay: Tensor, i_rel_decay: Tensor):
        
        if(self.edge_index_norm is None):
            self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
                  
        #u_emb_0 = self.users_emb.weight + self._u_abs_drift_emb.weight + self._u_rel_drift_emb.weight + self._u_base_emb.weight
        u_emb_0 = self.users_emb.weight
        
        if self.u_abs_drift:
            u_emb_0 = u_emb_0 + self._u_abs_drift_emb.weight
            
        if self.u_rel_drift:
            u_emb_0 = u_emb_0 + self._u_rel_drift_emb.weight
        
        if self.user_baseline:
            u_emb_0 = u_emb_0 + self._u_base_emb.weight
            
        #u_emb_0 = self.users_emb.weight #+ self._u_abs_drift_emb.weight + self._u_rel_drift_emb.weight
        i_emb_0 = self.items_emb.weight #+ self._i_base_emb.weight
        
        emb_0 = torch.cat([u_emb_0, i_emb_0])
        embs = [emb_0]
        emb_k = emb_0
        
        #if(self.edge_index_norm is None):
        #    # Compute normalization
        #    from_, to_ = edge_index
        #    deg = degree(to_, self.num_users + self.num_items, dtype=emb_k.dtype)
        #    deg_inv_sqrt = deg.pow(-0.5)
        #    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #    self.edge_index_norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
    
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index=self.edge_index_norm[0], x=emb_k, norm=self.edge_index_norm[1])
            embs.append(emb_k)
             
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)          
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        
        self.users_emb_final = users_emb_final
        self.items_emb_final = items_emb_final
        
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]
        
        _inner_pro = torch.mul(user_embeds, item_embeds)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[src]
            _inner_pro = _inner_pro + _u_base_emb
            
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[dest]
            _inner_pro = _inner_pro + _i_base_emb
            
        if self.u_abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[src]
            _u_abs_drift_emb = _u_abs_drift_emb * u_abs_decay.unsqueeze(1)
            _inner_pro = _inner_pro + _u_abs_drift_emb
            
        if self.u_rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[src]
            _u_rel_drift_emb = _u_rel_drift_emb * u_rel_decay.unsqueeze(1) 
            _inner_pro = _inner_pro + _u_rel_drift_emb
             
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.model != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
            ratings = self.f(_inner_pro)
        else:
            ratings = _inner_pro
        
        #ratings = self.f(_inner_pro)       
        return ratings
    
    def message(self, x_j, norm):
        out =  x_j * norm.view(-1, 1)
                
        return out
    
    #def aggregate(self, edge_index, x, norm, edge_weight=None):
    #    row, col = edge_index
    #    out = self.message(x[col], norm, edge_weight)
    #    out = torch_scatter.scatter_add(out, row, dim=0, dim_size=x.size(0))
    #    #  out = torch_scatter.scatter(inputs, index, dim=0, reduce='sum')
    #    return out
    
    def predict(self, u_id, i_id):
        
        user_embed = self.users_emb_final[u_id]
        item_embed = self.items_emb_final[i_id]
        
        _inner_pro = torch.mul(user_embed, item_embed)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[u_id]
            _inner_pro = _inner_pro + _u_base_emb
        
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[i_id]
            _inner_pro = _inner_pro + _i_base_emb
        
        if self.u_abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[u_id]
            #_u_abs_drift_emb = _u_abs_drift_emb * u_abs_t_decay.unsqueeze(1)
            _inner_pro = _inner_pro + _u_abs_drift_emb
            
        if self.u_rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[u_id]
            #_u_rel_drift_emb = _u_rel_drift_emb * u_rel_t_decay.unsqueeze(1) 
            _inner_pro = _inner_pro + _u_rel_drift_emb
             
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.model != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
        
        rating = self.f(_inner_pro)
              
        return rating

    def save_embeddings(self, path):
        torch.save({
            'users_emb_final': self.users_emb_final,
            'items_emb_final': self.items_emb_final
        }, path)

    def load_embeddings(self, path):
        checkpoint = torch.load(path)
        self.users_emb_final = checkpoint['users_emb_final']
        self.items_emb_final = checkpoint['items_emb_final']
        

# pred = u_b * i_b + (u_abs * u_abs_t_decay)*(i_abs*i_abs_t_decay) + u_rel * u_rel_t_decay + mu
class LGCN_full(MessagePassing):    
    def __init__(self, 
                 model,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 num_layers=0,
                 add_self_loops = False,
                 mu = 0,
                 drop= 0,
                 verbose = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.edge_index_norm = None
        self.verbose = verbose
        self.mu = mu
        self.model = model
        self.dropout = drop
        
        print("Model:", model, " | Layers:", num_layers, " | emb dimension:", embedding_dim, " | dropout:", drop)
        
        self.user_baseline = False
        self.item_baseline = False
        self.abs_drift = False
        self.rel_drift = False
        
        self.users_emb_final = None
        self.items_emb_final = None
        
        if model == 'lgcn_b':  # lgcn + baseline
            self.item_baseline = True
            self.user_baseline = True
        elif model == 'lgcn_b_a':  # lgcn + baseline + absolute
            self.abs_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif model == 'lgcn_b_r':  # lgcn + baseline + relative
            self.rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif model == 'lgcn_b_ar': # lgcn + baseline + absolute + relative 
            self.abs_drift = True
            self.rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif model == 'lgcn_ar': # lgcn + baseline + absolute + relative 
            self.abs_drift = True
            self.rel_drift = True
        else: # pure lightGCN model only
            model = 'lgcn'
            self.mu = 0
        
        
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        
        self.users_emb.weight.requires_grad = True
        self.items_emb.weight.requires_grad = True
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        if self.user_baseline:
            self._u_base_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim)
            nn.init.zeros_(self._u_base_emb.weight)
            self._u_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The user baseline embedding is ON.")
        
        if self.item_baseline:
            self._i_base_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim)
            nn.init.zeros_(self._i_base_emb.weight)
            self._i_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The item baseline embedding is ON.")

        if self.abs_drift:
            self._u_abs_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim)     
            nn.init.zeros_(self._u_abs_drift_emb.weight)
            self._u_abs_drift_emb.weight.requires_grad = True
            
            self._i_abs_drift_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim)     
            nn.init.zeros_(self._i_abs_drift_emb.weight)
            self._i_abs_drift_emb.weight.requires_grad = True
            
            if self.verbose:
                print("The absolute temporal embeddings are ON.")

        if self.rel_drift:
            self._u_rel_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim)     
            nn.init.zeros_(self._u_rel_drift_emb.weight)
            self._u_rel_drift_emb.weight.requires_grad = True
            
            self._i_rel_drift_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim)     
            nn.init.zeros_(self._i_rel_drift_emb.weight)
            self._i_rel_drift_emb.weight.requires_grad = True
            
            if self.verbose:
                print("The relative temporal embeddings are ON.")
                
        self.f = nn.ReLU()
        #self.f = nn.SiLU()
              
    def forward(self, edge_index: Tensor, src: Tensor, dest: Tensor, abs_decay: Tensor, u_rel_decay: Tensor, i_rel_decay: Tensor):
        
        if(self.edge_index_norm is None):
            self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
                  
        #u_emb_0 = self.users_emb.weight + self._u_abs_drift_emb.weight + self._u_rel_drift_emb.weight
        
        #u_emb_0 = self.users_emb.weight + self._u_abs_drift_emb.weight + self._u_rel_drift_emb.weight
        #i_emb_0 = self.items_emb.weight + self._i_abs_drift_emb.weight + self._i_rel_drift_emb.weight
        u_emb_0 = self.users_emb.weight + self._u_rel_drift_emb.weight
        i_emb_0 = self.items_emb.weight + self._i_rel_drift_emb.weight
        
        #u_emb_0 = self.users_emb.weight
        #i_emb_0 = self.items_emb.weight
         
        emb_0 = torch.cat([u_emb_0, i_emb_0])
        embs = [emb_0]
        emb_k = emb_0
        
        #if(self.edge_index_norm is None):
        #    # Compute normalization
        #    from_, to_ = edge_index
        #    deg = degree(to_, self.num_users + self.num_items, dtype=emb_k.dtype)
        #    deg_inv_sqrt = deg.pow(-0.5)
        #    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #    self.edge_index_norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index=self.edge_index_norm[0], x=emb_k, norm=self.edge_index_norm[1])
            embs.append(emb_k)
             
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)          
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        
        self.users_emb_final = users_emb_final
        self.items_emb_final = items_emb_final
        
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]
    
        #user_embeds = user_embeds * abs_decay.unsqueeze(1)
        #item_embeds = item_embeds * abs_decay.unsqueeze(1)
        
        _inner_pro = torch.mul(user_embeds, item_embeds)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[src]
            _inner_pro = _inner_pro + _u_base_emb
            
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[dest]
            _inner_pro = _inner_pro + _i_base_emb
            
        if self.abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[src]
            _u_abs_drift_emb = _u_abs_drift_emb * abs_decay.unsqueeze(1)
            
            _i_abs_drift_emb = self._i_abs_drift_emb.weight[dest]
            _i_abs_drift_emb = _i_abs_drift_emb * abs_decay.unsqueeze(1)
            
            abs_drift = torch.mul(_u_abs_drift_emb, _i_abs_drift_emb)

            _inner_pro = _inner_pro + self.f(abs_drift)
            #_inner_pro = _inner_pro + _u_abs_drift_emb + _i_abs_drift_emb
            #_inner_pro = _inner_pro + _u_abs_drift_emb
                
        if self.rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[src]
            _u_rel_drift_emb = _u_rel_drift_emb * u_rel_decay.unsqueeze(1) 
            
            _i_rel_drift_emb = self._i_rel_drift_emb.weight[dest]
            _i_rel_drift_emb = _i_rel_drift_emb * i_rel_decay.unsqueeze(1) 
            
            rel_drift = torch.mul(_u_rel_drift_emb, _i_rel_drift_emb)
            
            #_u_rel_drift_emb = self.f(_u_rel_drift_emb)
            _inner_pro = _inner_pro + self.f(rel_drift)
            #_inner_pro = _inner_pro + _u_rel_drift_emb + _i_rel_drift_emb
             
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.model != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
            ratings = self.f(_inner_pro)
        else:
            ratings = _inner_pro
        
        #ratings = self.f(_inner_pro)
              
        return ratings
    
    def message(self, x_j, norm):
        out =  x_j * norm.view(-1, 1)
                
        return out
    
    #def aggregate(self, edge_index, x, norm, edge_weight=None):
    #    row, col = edge_index
    #    out = self.message(x[col], norm, edge_weight)
    #    out = torch_scatter.scatter_add(out, row, dim=0, dim_size=x.size(0))
    #    #  out = torch_scatter.scatter(inputs, index, dim=0, reduce='sum')
    #    return out
    
    def predict(self, u_id, i_id):
        
        user_embed = self.users_emb_final[u_id]
        item_embed = self.items_emb_final[i_id]
        
        _inner_pro = torch.mul(user_embed, item_embed)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[u_id]
            _inner_pro = _inner_pro + _u_base_emb
        
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[i_id]
            _inner_pro = _inner_pro + _i_base_emb
        
        if self.u_abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[u_id]
            #_u_abs_drift_emb = _u_abs_drift_emb * u_abs_t_decay.unsqueeze(1)
            _inner_pro = _inner_pro + _u_abs_drift_emb
            
        if self.u_rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[u_id]
            #_u_rel_drift_emb = _u_rel_drift_emb * u_rel_t_decay.unsqueeze(1) 
            _inner_pro = _inner_pro + _u_rel_drift_emb
             
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.model != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
        
        rating = self.f(_inner_pro)
              
        return rating

    def save_embeddings(self, path):
        torch.save({
            'users_emb_final': self.users_emb_final,
            'items_emb_final': self.items_emb_final
        }, path)

    def load_embeddings(self, path):
        checkpoint = torch.load(path)
        self.users_emb_final = checkpoint['users_emb_final']
        self.items_emb_final = checkpoint['items_emb_final']
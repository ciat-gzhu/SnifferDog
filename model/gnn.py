import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch.optim.adamw import AdamW
from tqdm import trange, tqdm

from model.layer import EGADLayer


class EGAD(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_nodes: int,
                 num_layers: int, dropout: float, aggregate_type: str,
                 edge_feat: np.ndarray, edges: np.ndarray,
                 node_neighborNodes_dic: Dict[int, Set[int]], node_edge_dic: Dict[int, Set[int]],
                 device: torch.device, num_samples: int = 200, sparse: bool = True):
        super(EGAD, self).__init__()
        self.node_neighborNodes_dic = node_neighborNodes_dic
        self.edges = edges

        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.device = device

        node_feat_dim = hidden_dim
        edge_feat_dim = edge_feat.shape[1]
        self.node_embed = torch.ones(size=(num_nodes, node_feat_dim), dtype=torch.float32, device=device)
        self.np_edge_embed = torch.as_tensor(edge_feat)
        self.edge_embed = torch.asarray(edge_feat, dtype=torch.float32, device=device)

        # start_dim = 256
        gnn_out_dim = 4
        if num_layers == 1:
            out_dim = gnn_out_dim
        else:
            out_dim = 16

        # out_dim = gnn_out_dim = 128

        aggregate_type = aggregate_type.upper()
        node_attention = 'N2N' in aggregate_type
        edge_attention = 'N2E' in aggregate_type
        self.egad_layers = nn.Sequential(OrderedDict([
            ('egad_layer1',
             EGADLayer(self.node_embed, self.edge_embed, node_feat_dim, out_dim, edge_feat_dim, dropout,
                       node_attention, edge_attention,
                       node_edge_dic, node_neighborNodes_dic, device, self.thread_pool,
                       num_sample=num_samples, sparse=sparse))
        ]))
        self.last_layer = f'egad_layer{num_layers}'

        if num_layers > 1:
            last_layer = self.egad_layers.egad_layer1
            for layer_index in range(2, num_layers):
                input_dim = out_dim
                out_dim = gnn_out_dim
                last_layer = EGADLayer(last_layer, self.edge_embed, input_dim, out_dim, edge_feat_dim, dropout,
                                       node_attention, edge_attention,
                                       node_edge_dic, node_neighborNodes_dic, device, self.thread_pool,
                                       num_sample=num_samples, sparse=sparse)
                self.egad_layers.add_module(f'egad_layer{layer_index}', last_layer)

            input_dim = out_dim
            out_dim = gnn_out_dim
            self.egad_layers.add_module(self.last_layer,
                                        EGADLayer(last_layer, self.edge_embed, input_dim, out_dim, edge_feat_dim,
                                                  dropout, node_attention, edge_attention,
                                                  node_edge_dic, node_neighborNodes_dic, device, self.thread_pool,
                                                  num_sample=num_samples, sparse=sparse))

        # self.edge_ffn = pyg.nn.Linear(edge_feat_dim, gnn_out_dim, weight_initializer='glorot')
        # self.edge_attn = nn.MultiheadAttention(gnn_out_dim, num_heads=4, dropout=0.)
        # self.node_attn = nn.MultiheadAttention(gnn_out_dim, num_heads=4, dropout=0.)
        # self.node_ffn = pyg.nn.Linear(gnn_out_dim*2, gnn_out_dim, weight_initializer='glorot')


        # self.classifier = nn.Sequential(
        #     # nn.Dropout(dropout),
        #     pyg.nn.Linear(((2 * out_dim) + edge_feat_dim), num_classes, weight_initializer='glorot'),
        #     nn.LeakyReLU(inplace=True)
        # )
        self.final_edge_dim = (2 * gnn_out_dim) + edge_feat_dim
        self.edge_dim = edge_feat_dim
        self.node_dim = gnn_out_dim

        self.classifier = nn.Sequential(
            pyg.nn.Linear(((2 * out_dim) + edge_feat_dim), num_classes, weight_initializer='glorot'),
            # pyg.nn.Linear(self.final_edge_dim, 128, weight_initializer='glorot'),
            #
            # pyg.nn.BatchNorm(128),
            # nn.LeakyReLU(inplace=True),
            # pyg.nn.Linear(128, num_classes, weight_initializer='glorot')
        )

    def edge_embedding(self, edge_indices):
        unique_nodes = set()
        for u, v in self.edges[edge_indices]:
            unique_nodes.add(u)
            unique_nodes.add(v)
        unique_nodes = np.asarray(list(unique_nodes))

        node_embed = self.egad_layers._modules[self.last_layer](unique_nodes)

        nodeIndex_batchIndex_map = {node_index: batch_index for batch_index, node_index in enumerate(unique_nodes)}

        detach_node_embed = node_embed.clone().detach().cpu().numpy()
        edge_embed = np.asarray([
            np.concatenate(
                (
                    detach_node_embed[nodeIndex_batchIndex_map[self.edges[edge_index][0]]],
                    detach_node_embed[nodeIndex_batchIndex_map[self.edges[edge_index][1]]],
                    self.np_edge_embed[edge_index]
                )
            ) for edge_index in edge_indices
        ])
        edge_embed = torch.asarray(edge_embed, device=self.device, dtype=torch.float32, requires_grad=self.training)

        # edge_embed = []
        # start_node_embed = []
        # end_node_embed = []
        # for embed_index, edge_index in enumerate(edge_indices):
        #     start_node_index, end_node_index = self.edges[edge_index]
        #     start_node_embed.append(node_embed[nodeIndex_batchIndex_map[start_node_index]])
        #     end_node_embed.append(node_embed[nodeIndex_batchIndex_map[end_node_index]])
        #     edge_embed.append(self.np_edge_embed[edge_index].to(self.device))
        # edge_embed = torch.stack(edge_embed)
        # start_node_embed = torch.stack(start_node_embed)
        # end_node_embed = torch.stack(end_node_embed)
        # node_embed = torch.stack((start_node_embed, end_node_embed), dim=1)
        # node_embed, _ = self.node_attn(node_embed, node_embed, node_embed)
        # node_embed = self.node_ffn(node_embed.view(-1, self.node_dim*2))
        #
        # edge_embed = self.edge_ffn(edge_embed)
        # attn_edge_embed, _ = self.edge_attn(edge_embed, node_embed, node_embed)
        # edge_embed = self.layer_norm(edge_embed + attn_edge_embed)

        # detach_node_embed = node_embed.clone().detach().cpu().numpy()
        # if self.training:
        #     edge_embed = torch.stack(
        #         [
        #             torch.hstack(
        #                 (
        #                     node_embed[nodeIndex_batchIndex_map[self.edges[edge_index][0]]],
        #                     node_embed[nodeIndex_batchIndex_map[self.edges[edge_index][1]]],
        #                     self.np_edge_embed[edge_index].to(self.device)
        #                 )
        #             )
        #             for edge_index in edge_indices
        #         ]
        #     )
        # else:
        #     with torch.no_grad():
        #         node_embed = node_embed.detach().cpu()
        #         edge_embed = torch.empty((len(edge_indices), (2*self.node_dim)+self.edge_dim), dtype=torch.float32, requires_grad=False)
        #         for embed_index, edge_index in enumerate(edge_indices):
        #             start_node_index, end_node_index = self.edges[edge_index]
        #             edge_embed[embed_index, : self.node_dim] = node_embed[nodeIndex_batchIndex_map[start_node_index]]
        #             edge_embed[embed_index, self.node_dim: 2*self.node_dim] = node_embed[nodeIndex_batchIndex_map[end_node_index]]
        #             edge_embed[embed_index, 2*self.node_dim:] = self.np_edge_embed[edge_index]
        return edge_embed

    def forward(self, edge_indices):
        edge_embed = self.edge_embedding(edge_indices)
        return self.classifier(edge_embed)

    def garbage_collect(self):
        self.thread_pool = None
        self.np_edge_embed = None
        self.node_embed = None
        self.edge_embed = None
        self.edges = None
        self.node_neighborNodes_dic = None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def fit(self, train_loader, edge_label, num_epochs: int):
        self.train()

        # self.meta_learn(train_loader, 1)
        optimizer = AdamW(self.parameters(), lr=3e-4, weight_decay=3e-5)
        # classify_criterion = get_classify_criterion(loss_type, weight, self.device)

        for epoch in trange(num_epochs, desc='Fit'):
            epoch_progress = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}')
            for batch_edges in epoch_progress:
                optimizer.zero_grad()

                batch_edges = batch_edges[0].numpy()
                y_pred = self.forward(batch_edges)

                # loss = F.mse_loss(self.reconstruction_ffn(edge_embed).to(dtype=torch.float32), self.np_edge_embed[batch_edges].to(device=self.device, dtype=torch.float32))
                # if epoch != 1:
                batch_label = torch.as_tensor(edge_label[batch_edges], device=self.device, dtype=torch.long)
                loss = F.cross_entropy(y_pred, batch_label)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1, norm_type=2)
                optimizer.step()
                epoch_progress.set_postfix({'Loss': f'{loss.item():.4f}'})


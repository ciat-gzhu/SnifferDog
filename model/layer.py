import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Set, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg


def sample_and_mask(neighbors: List[Set[int]], num_sample: Optional[int],
                    attention_aggregate: bool, device: torch.device, sparse: bool = True)\
        -> Union[List[int], Tuple[List[int], Union[torch.Tensor, torch.sparse.Tensor]]]:
    # 以邻接表形式, 采样每个节点的邻居节点或相连边
    if num_sample is not None:
        sample_neighbors = [
            set(random.sample(neighbor, num_sample,))
            # 邻居数量少于采样数时, 取全部
            if len(neighbor) > num_sample else neighbor
            for neighbor in neighbors
        ]
    else:
        sample_neighbors = neighbors

    node_neighbors_list = list(set.union(*sample_neighbors))
    # 如果不使用注意力聚合, 则只采样不做Mask
    if not attention_aggregate:
        return node_neighbors_list

    # 生成邻接表形式, 格式为(num_nodes, num_neighbor_nodes)的Mask, 用于表示节点间的相连关系
    neighbor_neighborIndex_map = {neighbors: node_index for node_index, neighbors in enumerate(node_neighbors_list)}
    column_indices, row_indices = [], []
    for node_index, neighbor_nodesOrEdges in enumerate(sample_neighbors):
        for neighbor in neighbor_nodesOrEdges:
            column_indices.append(neighbor_neighborIndex_map[neighbor])
            row_indices.append(node_index)

    if sparse:
        indices = torch.vstack((torch.as_tensor(row_indices, device=device),
                               torch.as_tensor(column_indices, device=device)))
        mask = torch.sparse_coo_tensor(indices=indices, values=torch.ones(len(row_indices), dtype=torch.int8, device=device),
                                       size=(len(neighbors), len(node_neighbors_list)), dtype=torch.int8, device=device)
    else:
        mask = torch.zeros((len(neighbors), len(node_neighbors_list)), dtype=torch.int8, device=device)
        mask[row_indices, column_indices] = 1
    return node_neighbors_list, mask


class EGADLayer(nn.Module):
    def __init__(self, node_embed, edge_embed,
                 in_dim: int, out_dim: int, edge_feat_dim: int, dropout: float,
                 node_attention: bool, edge_attention: bool,
                 node_edge_dic: Dict[int, Set[int]], node_neighborNodes_dic: Dict[int, Set[int]],
                 device: torch.device, thread_pool: ThreadPoolExecutor, num_sample: int = None, sparse: bool = True):
        super(EGADLayer, self).__init__()

        self.node_feat_dim = in_dim

        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.node_edge_dic = node_edge_dic
        self.node_neighborNodes_dic = node_neighborNodes_dic
        self.num_sample = num_sample
        self.device = device

        self.thread_pool = thread_pool
        self.sparse = sparse

        self.node_matrix = None
        # self.no_neighbor_node_embed = torch.ones((1, self.node_feat_dim), dtype=torch.float32, device=self.device)
        self.node_attention = node_attention
        self.edge_attention = edge_attention
        self.W_v = nn.Parameter(torch.empty((edge_feat_dim, in_dim), dtype=torch.float32))
        self.W_e = nn.Parameter(torch.empty((in_dim, edge_feat_dim), dtype=torch.float32))
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_e)
        self.fc = nn.Sequential(
            pyg.nn.Linear(in_dim, out_dim, weight_initializer='glorot'),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def edge_message_propagate(self, nodes, neighbor_edges):
        result_sample_and_mask = self.thread_pool.submit(sample_and_mask, neighbor_edges, self.num_sample, self.edge_attention, self.device, self.sparse)
        if isinstance(self.node_embed, torch.Tensor):
            node_matrix = self.node_embed[nodes]
        else:
            node_matrix = self.node_embed(nodes)
        self.node_matrix = node_matrix.clone()
        node_matrix = torch.matmul(node_matrix, self.W_e)
        # print(f'Num Nodes: {len(nodes)}')
        # print(f'h_v shape: {node_matrix.size()}')

        if self.edge_attention:
            node_norms = torch.norm(node_matrix, dim=1, keepdim=True)
            unique_edges, mask = result_sample_and_mask.result()
            # Equation (13)
            edge_embed = torch.as_tensor(self.edge_embed[unique_edges], device=self.device)
            # print(f'h_D(v) shape: {edge_embed.size()}')
            edge_norms = torch.norm(edge_embed, dim=1, keepdim=True)

            cosine_similarities = torch.mm(node_matrix, edge_embed.t()) / torch.mm(node_norms, edge_norms.t())
            # print(f'Dist shape: {cosine_similarities.size()}, mask shape: {mask.size()}')
            if self.sparse:
                cosine_distance = torch.sparse.softmax(torch.mul((1 - cosine_similarities), mask), dim=1)
            else:
                cosine_distance = F.softmax((1 - cosine_similarities) * mask, dim=1)
            # print(f'Dist shape: {cosine_distance.size()},')

            # Equation (14)
            to_feats = torch.matmul(cosine_distance, edge_embed)
            # print(f'to_feats shape: {to_feats.size()}')
            # exit()
            embed_weight = torch.matmul(to_feats, self.W_v)

        else:
            unique_edges = set(result_sample_and_mask.result())
            edge_embed = torch.vstack([
                    F.adaptive_avg_pool1d(
                        self.edge_embed[[edge for edge in self.node_edge_dic[node] if edge in unique_edges]].transpose(0, 1),
                        output_size=1
                    ).transpose(0, 1)
                    for node in nodes
            ])
            embed_weight = torch.matmul((node_matrix + edge_embed), self.W_v)
        embed_weight[torch.isnan(embed_weight)] = 1e-2
        return embed_weight

    def node_message_propagate(self, nodes, neighbor_nodes):
        sample_res = sample_and_mask(neighbor_nodes, self.num_sample, self.node_attention, self.device, self.sparse)
        if self.node_attention:
            unique_nodes, mask = sample_res
            # 获取节点特征矩阵
            if isinstance(self.node_embed, torch.Tensor):
                neighbor_matrix = self.node_embed[unique_nodes]
            else:
                neighbor_matrix = self.node_embed(unique_nodes)
            if self.node_matrix is None:
                if isinstance(self.node_embed, torch.Tensor):
                    node_matrix = self.node_embed[nodes]
                else:
                    node_matrix = self.node_embed(nodes)
                self.node_matrix = node_matrix.clone()
            # 先计算每个节点与每条边的欧式距离
            distances = torch.cdist(self.node_matrix, neighbor_matrix, p=2).to(self.device)
            # 再乘上每个节点与每条边是否相邻的mask
            if self.sparse:
                neighbor_node_embeds = torch.sparse.softmax(torch.mul(distances, mask), dim=1).mm(neighbor_matrix)
            else:
                neighbor_node_embeds = F.softmax(distances * mask, dim=1).mm(neighbor_matrix)

        else:
            unique_nodes = set(sample_res)

            neighbor_node_embeds = []
            # 逐节点计算平均聚合
            for node in nodes:
                neighbor_nodes = list(node for node in self.node_neighborNodes_dic[node] if node in unique_nodes)
                # 如果当前节点没有邻居，就用一个全 1 向量代替
                if len(neighbor_nodes) == 0:
                    neighbor_node_embeds.append(
                        torch.ones((1, self.node_feat_dim), dtype=torch.float32, device=self.device)
                    )
                else:
                    # 对当前节点的邻居节点进行平均聚合
                    if isinstance(self.node_embed, torch.Tensor):
                        neighbor_node_embed = F.adaptive_avg_pool1d(self.node_embed[neighbor_nodes].transpose(0, 1), 1)
                    else:
                        neighbor_node_embed = F.adaptive_avg_pool1d(self.node_embed(neighbor_nodes).transpose(0, 1), 1)
                    neighbor_node_embeds.append(neighbor_node_embed.transpose(0, 1))
            neighbor_node_embeds = torch.vstack(neighbor_node_embeds)
        neighbor_node_embeds[torch.isnan(neighbor_node_embeds)] = 1e-2
        return neighbor_node_embeds

    def forward(self, nodes):
        self.node_matrix = None
        if self.edge_attention:
            neighbor_edges = [self.node_edge_dic[int(node)] for node in nodes]
            neighbor_edge_feat = self.edge_message_propagate(nodes, neighbor_edges)
        else:
            neighbor_edge_feat = None
        if self.node_attention:
            neighbor_nodes = [self.node_neighborNodes_dic[int(node)] for node in nodes]
            neighbor_node_feat = self.node_message_propagate(nodes, neighbor_nodes)
        else:
            neighbor_node_feat = None
        embeddings = self.node_matrix
        if neighbor_node_feat is not None:
            embeddings = embeddings + neighbor_node_feat
        if neighbor_edge_feat is not None:
            embeddings = embeddings + neighbor_edge_feat
        return self.fc(embeddings)

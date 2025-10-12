import time
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from os import cpu_count
from os.path import exists
from typing import Union

import numpy as np
import torch
from beeprint import pp
from sklearnex.model_selection import train_test_split
from tqdm import trange

from flow_encoder.context_builder import ContextBuilder
from utils.eval import calc_score
from utils.file import load_pickle


def calc_class_weight(label_count):
    total_count = sum(label_count)
    alpha = [total_count / (num * len(label_count)) for num in label_count]
    sum_alpha = sum(alpha)
    alpha = torch.as_tensor([a / sum_alpha for a in alpha], dtype=torch.float32)  # 归一化
    return alpha


def stat_class_ratio(labels):
    # 统计每个类别出现的次数
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # 计算每个类别的占比
    label_proportions = label_counts / len(labels)

    # 将类别和对应的占比打印出来
    for label, count, proportion in zip(unique_labels, label_counts, label_proportions):
        print(f"Class {label}, count: {count}, ratio: {proportion:.2%}")
    return label_counts


def load_new_unsw(binary_classify: bool):
    dataset_path = f'./datasets/new_UNSW-NB15'
    edge_feat = np.load(f'{dataset_path}/data.npy')
    edge_label = np.load(f'{dataset_path}/label.npy')
    stat_class_ratio(edge_label)
    if binary_classify:
        edge_label[np.where(edge_label != 0)[0]] = 1

    train_x, test_x, train_y, test_y = train_test_split(edge_feat, edge_label, test_size=0.7, random_state=42)

    from sklearnex.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf = clf.fit(train_x, train_y)
    random_classifier_proba = clf.predict_proba(test_x).astype(np.float32)
    random_classifier_res = calc_score(random_classifier_proba, test_y, binary_classify)
    pp(random_classifier_res)

    edges = np.load(f'{dataset_path}/edges.npy')
    node_edge_dic = torch.load(f'{dataset_path}/node_edge_dic.pt')
    node_neighborNodes_dic = torch.load(f'{dataset_path}/node_neighborNodes_dic.pt')

    return {
        'weight': calc_class_weight(stat_class_ratio(edge_label)),
        'num_nodes': len(np.unique(edges)),
        "edge_feat": edge_feat,  # shape=(E, G_e)
        "edge_index": edges,  # shape=(E, 2)
        "edge_label": edge_label,  # 边标签集, shape=(E)
        "node_edge_dic": node_edge_dic,  # 各节点连接的边, 以Dict[int, Set[int]]格式存储
        "node_neighborNodes_dic": node_neighborNodes_dic  # 各节点的邻居节点, 以Dict[int, Set[int]]格式存储
    }


def load_iot_data(dataset, binary_classify):
    dataset_prefix = f'./datasets/{dataset}'
    edge_feat = np.load(f'{dataset_prefix}/edge_embeddings.npy')
    edge_label = np.load(f'{dataset_prefix}/edge_labels.npy')
    if binary_classify:
        edge_label[np.where(edge_label != 0)[0]] = 1

    node_edge_dic = load_pickle(f'{dataset_prefix}/node_edge_dic.pkl')
    node_neighborNodes_dic = load_pickle(f'{dataset_prefix}/node_neighbor_dic.pkl')

    return {
        'weight': calc_class_weight(stat_class_ratio(edge_label)),
        'num_nodes': len(load_pickle(f'{dataset_prefix}/node_nodeIndex_dic.pkl')),
        "edge_feat": edge_feat,  # shape=(E, G_e)
        "edge_index": np.load(f'{dataset_prefix}/edge_index.npy'),  # shape=(E, 2)
        "edge_label": edge_label,  # 边标签集, shape=(E)
        "node_edge_dic": node_edge_dic,  # 各节点连接的边, 以Dict[int, Set[int]]格式存储
        "node_neighborNodes_dic": node_neighborNodes_dic  # 各节点的邻居节点, 以Dict[int, Set[int]]格式存储
    }


def load_data(dataset: str, stream_aggregate: bool, stat_feat: bool,
              binary_classify: bool, device: torch.device,
              need_edge_feat: bool = True, need_graph: bool = True,
              flow_attention: bool = True, flow_rnn: bool = True,
              feat_visualize: bool = False, random_seed: int = 42):
    if dataset == 'diy' or dataset == 'diy_pool':
        return load_diy_data(dataset, binary_classify, stream_aggregate, stat_feat)
    elif dataset =='CICIOT2023' or dataset == 'CICIOT2023_pool':
        return load_iot_data(dataset, binary_classify)
    elif dataset == 'new_UNSW-NB15':
        return load_new_unsw(binary_classify)

    data_path_prefix = f"./datasets/{dataset}"

    with ThreadPoolExecutor(max_workers=cpu_count()) as thread_pool:
        label_file = 'label_bi' if binary_classify else 'label_mul'
        edge_label = np.load(f'{data_path_prefix}/{label_file}.npy', allow_pickle=True).astype(np.int8)

        label_counts = stat_class_ratio(edge_label)

        if need_edge_feat:
            processed_stream_feat_path = f'{data_path_prefix}/stream_feat_scaled_processed.npy'
            raw_stream_feat_path = f'{data_path_prefix}/stream_feat.npy'
            do_stream_aggregate = stream_aggregate and (exists(processed_stream_feat_path) or exists(raw_stream_feat_path))
            if do_stream_aggregate:
                packet_length = 148 - 12
                num_window_packets = 5
                if exists(processed_stream_feat_path):
                    edge_feat = np.load(processed_stream_feat_path)
                else:
                    stream_feat = np.load(raw_stream_feat_path)
                    # stream_feat = torch.as_tensor(stream_feat).view(-1, 5*148)
                    # stream_feat = z_score_normalize(stream_feat).view(-1, 5, 148)
                    packet_list = []
                    stream_feat = torch.as_tensor(stream_feat).reshape(-1, 5, 148)
                    for packet_index in trange(len(stream_feat), desc='Process Data'):
                        window = stream_feat[packet_index]
                        for _packet_index in range(5):
                            packet = window[_packet_index]
                            # IF TCP
                            if packet[60:80].sum().item() > 0:
                                packet = torch.cat((packet[:12], packet[20:60], packet[64:]))
                            else:
                                packet = torch.cat((packet[:12], packet[20:120], packet[124:]))
                            packet_list.append(packet)
                    stream_feat = torch.vstack(packet_list).to(device)
                    stream_feat = z_score_normalize(stream_feat).cpu().reshape(-1, num_window_packets*packet_length).numpy()
                    # np.save(processed_stream_feat_path, stream_feat)
                    X, y = stream_feat[:, :-packet_length], stream_feat[:, -packet_length:]
                    max_length = num_window_packets - 1

                    context_builder = ContextBuilder(
                        input_size=packet_length,  # Number of input features to expect
                        output_size=packet_length,  # Same as input size
                        hidden_size=128,  # Number of nodes in hidden layer, in paper we set this to 128
                        max_length=max_length,  # Length of the context, should be same as context in Preprocessor
                        device=device,
                        bidirectional=False,
                        attention=flow_attention,
                        rnn=flow_rnn
                    ).to(device)

                    window_num = packet_length * max_length
                    flows = None
                    for start in range(0, X.shape[1], window_num):
                        # Train the ContextBuilder and then do prediction
                        flow = context_builder.fit_predict(
                            X=X[:, start: start + window_num],  # Context to train with
                            y=y,  # Events to train with, note that these should be of shape=(n_events, 1)
                            epochs=5,  # Number of epochs to train with
                            batch_size=256,  # Number of samples in each training batch, in paper this was 128
                            learning_rate=1e-2,  # Learning rate to train with, in paper this was 0.01
                            verbose=True,  # If True, prints progress
                        ).detach()

                        if start == 0:
                            flows = flow
                        else:
                            flows += flow
                    edge_feat = torch.div(flows, window_num).cpu().numpy()
                    np.save(processed_stream_feat_path, edge_feat)

                if stat_feat:
                    stat_feat = np.load(f'{data_path_prefix}/edge_feat_scaled.npy').astype(np.float32)
                    edge_feat = np.hstack((z_score_normalize(edge_feat), stat_feat))
            else:
                edge_feat = z_score_normalize(np.load(f'{data_path_prefix}/edge_feat_scaled.npy').astype(np.float32))
        else:
            edge_feat = np.ones((len(edge_label), 1), dtype=np.float32)

        if need_graph:
            nodes = np.load(f'{data_path_prefix}/nodes.npy', allow_pickle=True)
            node_nodeIndex_map = {node: node_index for node_index, node in enumerate(nodes)}

            coo_adj = np.load(f'{data_path_prefix}/adj.npy', allow_pickle=True)

            node_edge_dic = defaultdict(set)  # Dict[Set[int]]格式存储的各节点连接的边
            node_neighborNodes_dic = defaultdict(set)  # Dict[Set[int]]格式存储的各节点邻居
            edges = np.empty(shape=(len(coo_adj), 2), dtype=np.int32)  # shape=(E, 2)的边集

            for edge_index, (node1, node2) in enumerate(coo_adj):
                nodeIndex1, nodeIndex2 = node_nodeIndex_map[node1], node_nodeIndex_map[node2]
                node_edge_dic[nodeIndex1].add(edge_index)
                node_edge_dic[nodeIndex2].add(edge_index)
                node_neighborNodes_dic[nodeIndex1].add(nodeIndex2)
                edges[edge_index][0] = nodeIndex1
                edges[edge_index][1] = nodeIndex2

        if feat_visualize:
            from utils.visual import edge_feat_visualize
            edge_feat_visualize(edge_feat, dataset, random_seed)

        if need_graph:
            return {
                "weight": calc_class_weight(label_counts),
                "num_nodes": len(node_nodeIndex_map),
                "edge_feat": edge_feat,  # shape=(E, G_e)
                "edge_index": edges.astype(np.int32),  # shape=(E, 2)
                "edge_label": edge_label,  # 边标签集, shape=(E)
                "node_edge_dic": node_edge_dic,  # 各节点连接的边, 以Dict[int, Set[int]]格式存储
                "node_neighborNodes_dic": node_neighborNodes_dic  # 各节点的邻居节点, 以Dict[int, Set[int]]格式存储
            }
        return {
            "edge_feat": edge_feat,  # shape=(E, G_e)
            "edge_label": edge_label,  # 边标签集, shape=(E)
        }


def load_diy_data(dataset: str, binary_classify: bool, need_edge_feat: bool, need_stat_feat: bool):
    # dataset_prefix = f'./datasets/{dataset}'
    dataset_prefix = f'./datasets/{dataset}_bak'

    # labels = np.load(f'{dataset_prefix}/edge_labels.npy')
    # benign_indices = np.where(labels == 0)[0]
    # malicious_indices = np.where(labels != 0)[0]
    # malicious_count = len(malicious_indices)
    # benign_count = len(labels) - malicious_count
    #
    # goal_malicious_count = int(35./65. * benign_count)
    # malicious_split_ratio = float(goal_malicious_count) / float(malicious_count)
    #
    # malicious_indices, _ = train_test_split(malicious_indices, train_size=malicious_split_ratio,
    #                                         random_state=42, shuffle=True, stratify=labels[malicious_indices])
    # data_indices = np.concatenate((benign_indices, malicious_indices))
    # labels = labels[data_indices]
    # np.save('./output/DIY_index_65.npy', data_indices)
    # np.save('./output/DIY_label_65.npy', labels)
    #
    # edge_data = np.load(f'{dataset_prefix}/edge_embeddings.npy')[data_indices]
    # np.save('./output/DIY_F_65.npy', manifold(edge_data, 18))
    # np.save('./output/DIY_F_umap_65.npy', manifold(edge_data, 18, True))
    # stat_data = np.load(f'{dataset_prefix}/stat_embeddings_scaled.npy')[data_indices]
    # np.save('./output/DIY_S_65.npy', manifold(stat_data, seed=18, umap=True))
    # np.save('./output/DIY_S_umap_65.npy', manifold(stat_data, seed=18, umap=True))
    # exit()

    # np.save('./output/DIY_F.npy', manifold(np.load(f'{dataset_prefix}/edge_embeddings.npy'), 18))
    # np.save('./output/DIY_S_umap.npy', manifold(np.load(f'{dataset_prefix}/stat_embeddings_scaled.npy'), seed=18, umap=True))
    # exit()

    node_edge_dic = load_pickle(f'{dataset_prefix}/node_edge_dic.pkl')
    node_edge_dic = {k: set(v) for k, v in node_edge_dic.items()}

    node_neighborNodes_dic = load_pickle(f'{dataset_prefix}/node_neighbor_dic.pkl')
    # node_edge_dic = {node: set(edge_set) for node, edge_set in node_edge_dic.items()}
    node_neighborNodes_dic = {node: set(neighbors) for node, neighbors in node_neighborNodes_dic.items()}
    edge_label = np.load(f'{dataset_prefix}/edge_labels.npy')
    if binary_classify:
        edge_label[np.where(edge_label != 0)[0]] = 1
    if need_edge_feat:
        data = edge_feat = np.load(f'{dataset_prefix}/edge_embeddings.npy')
    if need_stat_feat:
        data = stat_feat = np.load(f'{dataset_prefix}/stat_embeddings_scaled.npy')
    if need_edge_feat and need_stat_feat:
        data = np.concatenate((edge_feat, stat_feat), axis=1)
    assert len(data) == len(edge_label)
    print(f'Feat Dim: {data.shape[1]}, num_classes: {len(np.unique(edge_label))}')
    # exit()
    # print(edge_feat.shape, stat_feat.shape, edge_label.shape)
    # exit()

    # from sklearnex.ensemble import RandomForestClassifier
    # classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    # from sklearnex.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(data, edge_label, test_size=0.3)
    # classifier = classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print(calc_score(y_pred, y_test, binary_classify))
    # exit()

    return {
        'weight': calc_class_weight(stat_class_ratio(edge_label)),
        'num_nodes': len(load_pickle(f'{dataset_prefix}/node_nodeIndex_dic.pkl')),
        "edge_feat": data,  # shape=(E, G_e)
        "edge_index": np.load(f'{dataset_prefix}/edge_index.npy'),  # shape=(E, 2)
        "edge_label": edge_label,  # 边标签集, shape=(E)
        "node_edge_dic": node_edge_dic,  # 各节点连接的边, 以Dict[int, Set[int]]格式存储
        "node_neighborNodes_dic": node_neighborNodes_dic  # 各节点的邻居节点, 以Dict[int, Set[int]]格式存储
    }


# 数据Z-score标准化
def z_score_normalize(data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(data, torch.Tensor):
        mean = torch.nanmean(data, dim=0)
        std = torch.nan_to_num(torch.std(data, dim=0))
        std[std == 0] = 1e-8
        return torch.nan_to_num((data - mean) / std, nan=0., posinf=0., neginf=0.)
    else:
        # 计算每行数据的均值和标准差
        mean = np.nanmean(data, axis=1, keepdims=True)
        std = np.nanstd(data, axis=1, keepdims=True)
        # 将标准差为0的值替换为一个很小的值, 避免除零错误
        std[std == 0] = 1e-8
        # 对每行数据进行Z-score标准化
        return np.nan_to_num((data - mean) / std)


def load_data_and_process(dataset: str, stream_aggregate: bool, stat_feat: bool,
                          binary_classify: bool, train_size: float, device: torch.device,
                          need_edge_feat: bool = True, need_graph: bool = True,
                          flow_attention: bool = True, flow_rnn: bool = True,
                          feat_visualize: bool = False, random_seed: int = 42):
    time_start = time.time()
    print(f'Load Data...', end='\r')
    data = load_data(dataset,
                     stream_aggregate, stat_feat, binary_classify, device,
                     need_edge_feat=need_edge_feat, need_graph=need_graph,
                     flow_attention=flow_attention, flow_rnn=flow_rnn,
                     feat_visualize=feat_visualize, random_seed=random_seed)

    if need_graph:
        edge_label = data["edge_label"]
        train_edges, test_edges = train_test_split(np.arange(len(edge_label)), train_size=train_size,
                                                   stratify=edge_label, random_state=random_seed)
        data['train_edges'] = train_edges
        data['test_edges'] = test_edges
    else:
        edge_embeds, edge_label = data['edge_feat'], data['edge_label']
        train_embeds, test_embeds, y_train, y_test = train_test_split(edge_embeds, edge_label,
                                                                      train_size=train_size, stratify=edge_label,
                                                                      random_state=random_seed)
        data = {
            'train_embeds': train_embeds,
            'test_embeds': test_embeds,
            'y_train': y_train,
            'y_test': y_test,
        }
    print(f'Load Data Done({time.time() - time_start:.1f} s).')
    return data


# 加载json数据
def load_json_data(file_path: str):
    from orjson import loads
    with open(file_path, "r") as json_file:
        json_data = json_file.read()
    return loads(json_data.replace(',]', ']').replace('][', '],['))

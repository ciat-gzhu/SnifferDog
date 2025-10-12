import gc
import os
import time

import numpy as np
import torch
from beeprint import pp as pretty_print
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import EGAD
from utils.eval import model_predict, calc_score
from utils.visual import visualize, plot_confusion_matrix


def save_model(model, model_path: str):
    os.makedirs(model_path.rsplit('/', 1)[0], exist_ok=True)
    torch.save(model.state_dict(), model_path)


def train(args, data, device: torch.device, num_classes: int, random_seed: int = 18, memory_track: bool = False):
    pretty_print(args)
    args.time = time.time()

    if args.graph:
        task = 'bin' if args.binary else 'mul'
        train_edges, test_edges = data['train_edges'], data['test_edges']
        edge_label = data['edge_label']

        hidden_dim = 128
        torch.cuda.empty_cache()
        model = EGAD(hidden_dim, num_classes, data["num_nodes"], args.num_layers, args.dropout,
                     args.aggregate_type,
                     data["edge_feat"], data["edge_index"], data['node_neighborNodes_dic'], data["node_edge_dic"],
                     device, args.num_samples, args.sparse).to(device)

        if args.isStreamAgg:
            if args.edge_feat:
                edge_feat_str = '_SFT'
            else:
                edge_feat_str = ''
        else:
            edge_feat_str = '_ST'
        model_path = f'./models/{args.dataset}_l{args.num_layers}{edge_feat_str}_{args.aggregate_type}_s{args.num_samples}_e{args.epochs}_{task}.pt'

        time_start = time.time()
        model = train_models(model, args.lr, train_edges, edge_label, device, args.epochs, args.train_batch_size)

        if args.save:
            save_model(model, model_path)
        print('-------------Train Done: {:.1f}s------------------'.format(time.time() - time_start))

        test_time1 = time.time()
        edge_embeds = model_predict(model, np.arange(len(edge_label)), args.eval_batch_size,
                                    verbose=1)
        test_time1 = args.train_size * (time.time() - test_time1)
        if args.graph_visual:
            from utils.visual import graph_visualize
            graph_visualize(edge_embeds, args.dataset, random_seed, args.isStreamAgg, args.stat_feat)
        train_embeds = edge_embeds[train_edges]
        test_embeds = edge_embeds[test_edges]

        # Train RandomForest Classifier
        rf_classifier = train_random_forest_classifier(train_embeds, edge_label[train_edges], random_seed)
        random_classifier_proba = rf_classifier.predict_proba(test_embeds)

        print(f'Test time: {test_time1}')

        y_eval = edge_label[test_edges]

        model.garbage_collect()
        del edge_embeds, train_embeds, test_embeds

    else:
        train_embeds, test_embeds, y_train, y_eval = data['train_embeds'], data['test_embeds'], data['y_train'], data['y_test']
        rf_classifier = train_random_forest_classifier(train_embeds, y_train, random_seed)
        random_classifier_proba = rf_classifier.predict_proba(test_embeds).astype(np.float32)

        del train_embeds, test_embeds, y_train

    random_classifier_proba = np.nan_to_num(random_classifier_proba, nan=1e-3, posinf=1e+3, neginf=1e-3)
    # if args.dataset in {'diy', 'diy_pool', 'CICIOT2023', 'CICIOT2023_pool'}:
    try:
        plot_confusion_matrix(y_eval, random_classifier_proba, args)
    except:
        pass
    random_classifier_res = calc_score(random_classifier_proba, y_eval, args.binary)
    pretty_print(random_classifier_res)

    if args.visualize:
        visualize(random_classifier_proba, y_eval, args.dataset, args.binary)

    # Numpy Garbage Collection
    del y_eval, random_classifier_proba, data
    if args.graph:
        del edge_label, train_edges, test_edges
    gc.collect()

    return random_classifier_res


def train_models(model, learning_rate: float,
                 train_edges: np.ndarray, label: np.ndarray, device: torch.device,
                 num_epochs: int, batch_size: int):
    from torch import nn, optim
    import torch.nn.functional as F
    train_edges = DataLoader(TensorDataset(torch.as_tensor(train_edges)), batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_progress = tqdm(train_edges, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in epoch_progress:
            optimizer.zero_grad()

            batch_edges = batch[0].numpy()
            scores = model(batch_edges)

            loss = F.cross_entropy(scores, torch.as_tensor(label[batch_edges], dtype=torch.long, device=device))
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            loss.backward()
            optimizer.step()

            epoch_progress.set_postfix({'loss': '{:.4f}'.format(loss.item())})
            epoch_progress.update(1)
    return model


def train_random_forest_classifier(edge_embeds, edge_labels, seed: int):
    from sklearnex.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=seed)
    rf_classifier.fit(edge_embeds, edge_labels)
    return rf_classifier

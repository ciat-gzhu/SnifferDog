from typing import Dict

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


@torch.no_grad()
def model_predict(model, edge_index: np.ndarray, batch_size: int, verbose: int = 0) -> np.ndarray:
    model.eval()

    embeds_feat = []
    loader = DataLoader(TensorDataset(torch.as_tensor(edge_index)), batch_size=batch_size, shuffle=False)

    for batch_edges in tqdm(loader, total=len(loader), disable=(verbose == 0), desc='Predict'):
        batch_edges = batch_edges[0].numpy()
        batch_pred = model.edge_embedding(batch_edges)
        if isinstance(batch_pred, torch.Tensor):
            batch_pred = batch_pred.detach().cpu()#.numpy()
        embeds_feat.append(batch_pred)
    if isinstance(embeds_feat[0], torch.Tensor):
        embeds_feat = torch.cat(embeds_feat).numpy()
    else:
        embeds_feat = np.concatenate(embeds_feat)
    return embeds_feat


def calc_score(y_prob: np.ndarray, y_eval: np.ndarray, binary: bool) -> Dict[str, float]:
    y_pred = y_prob.argmax(axis=1) if len(y_prob.shape) > 1 else y_prob
    conf_matrix = confusion_matrix(y_eval, y_pred)
    if binary:
        average = 'binary'
        tn, fp, fn, tp = conf_matrix.ravel()
    else:
        average = 'macro'

        tp = conf_matrix[0, 0]
        fn = np.sum(conf_matrix[0, :]) - tp
        fp = np.sum(conf_matrix[:, 0]) - tp
        tn = np.sum(conf_matrix) - tp - fn - fp
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    mcc = matthews_corrcoef(y_eval, y_pred)

    return {
        'acc': f1_score(y_eval, y_pred, average='micro', zero_division=1.0),
        # 'ap': ap,
        'pre-Mac': precision_score(y_eval, y_pred, average=average, zero_division=1.0),
        'pre-Wei': precision_score(y_eval, y_pred, average='weighted', zero_division=1.0),
        'rec-Mac': recall_score(y_eval, y_pred, average=average, zero_division=1.0),
        'rec-Wei': recall_score(y_eval, y_pred, average='weighted', zero_division=1.0),
        'f1-Mac': f1_score(y_eval, y_pred, average=average, zero_division=1.0),
        'f1-Wei': f1_score(y_eval, y_pred, average='weighted', zero_division=1.0),
        'binary': binary,
        'cm': conf_matrix,
        'mcc': mcc,
        # 'roc': roc,
        'tpr': tpr,
        'fpr': fpr,
    }

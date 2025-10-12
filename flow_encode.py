import os
from os.path import exists

import numpy as np
import torch
from tqdm import trange

from flow_encoder import ContextBuilder
from utils.visual import edge_feat_visualize


def z_score_normalize(data: torch.Tensor) -> torch.Tensor:
    mean = torch.nanmean(data, dim=0)
    std = torch.nan_to_num(torch.std(data, dim=0))
    std[std == 0] = 1e-6
    return torch.nan_to_num((data - mean) / std, nan=0., posinf=0., neginf=0.)


def process_data(dataset_name: str, save_path: str):
    raw_stream_feat_path = f'./datasets/{dataset_name}/stream_feat.npy'
    if not exists(raw_stream_feat_path):
        raw_stream_feat_path = f'./datasets/Unused/{dataset_name}/stream_feat.npy'

    stream_feat = np.load(raw_stream_feat_path)
    stream_feat = torch.as_tensor(stream_feat).view(-1, num_window_packets, 148)
    packet_list = []
    for packet_index in trange(len(stream_feat), desc='Process Data'):
        window = stream_feat[packet_index]
        window_list = []
        for _packet_index in range(5):
            packet = window[_packet_index]
            if packet[0].item() == 0:
                packet = packet[:packet_length]
            elif packet[20:60].sum().item() > 0:
                packet = torch.zeros(packet_length)
            elif packet[60:80].sum().item() > 0:
                packet = torch.cat((packet[:12], packet[64:]))
            else:
                packet = torch.cat((packet[:12], packet[60:120], packet[124:]))
            window_list.append(packet)
        packet_list.append(torch.cat(window_list, dim=0))
    stream_feat = torch.vstack(packet_list).to(device).view(-1, num_window_packets, 96)
    stream_feat = z_score_normalize(stream_feat).cpu().view(-1, num_window_packets * 96)
    torch.save(stream_feat, save_path)


if __name__ == '__main__':
    dataset_name = 'new_CICIDS2017'
    encode_flow_path = f'./datasets/{dataset_name}/encoded_flows.pt'
    process_data_path = f'./datasets/{dataset_name}/stream_feat_z.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    packet_length = 96
    num_window_packets = 5

    if not os.path.exists(encode_flow_path):
        if not os.path.exists(process_data_path):
            process_data(dataset_name, process_data_path)
        data = torch.load(process_data_path)

        model = ContextBuilder(input_size=packet_length, output_size=packet_length, hidden_size=128,
                               max_length=num_window_packets - 1, bidirectional=False, LSTM=False, device=device).to(
            device=device)
        data, label = data[:, :-packet_length], data[:, -packet_length:]
        encoded_flows = model.fit_predict(data, label, epochs=5, learning_rate=1e-3, batch_size=256).cpu()
        torch.save(encoded_flows, encode_flow_path)
    else:
        encoded_flows = torch.load(encode_flow_path)

    encoded_flows = encoded_flows.numpy()
    labels = np.load(f'./datasets/{dataset_name}/label_mul.npy')
    edge_feat_visualize(encoded_flows, labels, dataset_name, binary=False, random_seed=42)

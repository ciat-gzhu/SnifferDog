import os
import time
from os import makedirs
from typing import List, Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.file import load_pickle


def visualize(y_pred: np.ndarray, y_eval: np.ndarray, dataset: str, binary: bool = True):
    from pandas import crosstab
    df_cm = crosstab(y_eval, y_pred,
                        rownames=['Actual'], colnames=['Predicted'], dropna=False,
                        margins=False, normalize='index').round(4)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)

    from os import makedirs
    makedirs('./visual', exist_ok=True)
    plt.savefig(f"./visual/result_{dataset}_{binary}.pdf", bbox_inches='tight')


# 混淆矩阵可视化
def plot_confusion_matrix(y_true, y_pred, args, title=None, thresh=0.8, axis_labels=None):
    task = 'bin' if args.binary else 'mul'
    dataset = args.dataset
    dataset_prefix = dataset.rsplit('_', 1)[0]

    save_dir = f'./visualize/confusion_matrix/{dataset}'
    makedirs(save_dir, exist_ok=True)

    np.savez(
        f'{save_dir}/{task}_{args.aggregate_type}_l{args.num_layers}_e{args.epochs}_s{args.num_samples}.npz',
        **{'y_pred': y_pred, 'y_true': y_true})

    from sklearnex import patch_sklearn
    patch_sklearn()
    from sklearn.metrics import confusion_matrix
    labels_name = list(load_pickle(f'./datasets/{dataset_prefix}/label_map.pkl').values())\
        if not args.binary else ['Benign', 'Malicious']

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_name))), sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    # cm = np.nan_to_num(cm, nan=-5e-3)

    plt.figure(figsize=(15, 15))
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    heatmap = plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar(heatmap, fraction=0.045)  # 绘制图例

    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=60)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                         ha="center", va="center", fontsize=9,
                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    plt.tight_layout()  # 自动调整子图参数

    plt.savefig(f'{save_dir}/{task}_{args.aggregate_type}_l{args.num_layers}_e{args.epochs}_s{args.num_samples}.png', dpi=1000, bbox_inches='tight')


def edge_feat_visualize(edge_feat: np.ndarray, dataset: str, random_seed: int = 18):


    dataset_name = dataset.split('_', 1)[-1]
    # draw_legend(dataset_name)
    # visualize_flow_encode(x, y, title, title, random_seed)
    os.makedirs('./output', exist_ok=True)
    for manifold_method in [None, 'tsne', 'umap']:
        if manifold_method is None:
            np.save(f'./output/{dataset_name}_F_raw.npy', edge_feat)
        else:
            do_umap = manifold_method == 'umap'
            manifold_str = '_umap' if do_umap else ''
            np.save(f'./output/{dataset_name}_F{manifold_str}.npy', manifold(edge_feat, random_seed, do_umap))

    stat_data = np.load(f'./datasets/{dataset}/edge_feat_scaled.npy')
    for manifold_method in [None, 'tsne', 'umap']:
        if manifold_method is None:
            np.save(f'./output/{dataset_name}_S_raw.npy', edge_feat)
        else:
            do_umap = manifold_method == 'umap'
            manifold_str = '_umap' if do_umap else ''
            np.save(f'./output/{dataset_name}_S{manifold_str}.npy', manifold(edge_feat, random_seed, do_umap))
    exit()
    visualize_flow_encode(stat_data, y, title, title, random_seed)


def get_label_map(dataset_name):
    if dataset_name.startswith('CICIDS2017'):
        return {0: 'Benign', 1: 'Bot', 2: "Port Scan", 3: "DoS Hulk", 4: "SSH Brute", 5: "FTP Brute",
                6: "DoS GoldenEye"}
    elif dataset_name.upper().startswith('DIY'):
        return {0: 'benign', 1: 'tcp_dos', 2: 'ssh', 3: 'udp_dos', 4: 'ftp', 5: 'http_dos', 6: 'rdp',
                7: 'wordpress', 8: 'phpmyadmin', 9: 'syn_dos', 10: 'arp_mitm', 11: 'ssdp_flood',
                12: 'SSL_Renegotiation', 13: 'smb', 14: 'telnet', 15: 'webshell', 16: 'Fuzzing', 17: 'drupal', 18: 'mysql'}
    else:
        return {0: 'Benign', 1: 'Infiltrating', 2: 'HTTP DoS', 3: 'DDoS'}


def draw_legend(dataset_name):
    # --- 单独绘制图例 ---
    start_time = time.time()
    print(f'Build Legend Chart...', end='\r')
    label_map = get_label_map(dataset_name)

    # 创建一个新的 figure 和 axes 用于绘制图例
    # 调整 figsize 以适应图例的大小，可能需要一些尝试
    legend_figure, legend_ax = plt.subplots(figsize=(6, 6))  # 示例尺寸，根据您的标签数量调整

    unique_labels = sorted(set(label_map.values()))
    palette = sns.color_palette("colorblind", len(unique_labels))

    # 重新创建 handles，使用您现有的 label_map 和 palette
    # 确保 unique_labels 和 palette 的顺序与 label_map 的值对应
    # 这里假设 unique_labels 是 sorted(set(label_map.keys())) 或者 sorted(set(y)) 并且与 palette 颜色一一对应
    unique_labels_keys = sorted(set(label_map.keys()))  # 获取 label_map 的键并排序以匹配 palette 索引
    handles = [plt.Line2D([], [], marker="o", markersize=8, linestyle="", color=palette[i],
                          label=label_map[unique_labels_keys[i]])
               for i in range(len(unique_labels_keys))]

    # 在新的 axes 上绘制图例
    plt.legend(handles=handles, title='Label', loc='center', fontsize=14, title_fontsize=14, frameon=False)  # frameon=False 移除图例边框

    # 关闭图例图的坐标轴和边框
    plt.axis('off')
    legend_figure.patch.set_visible(False)  # 使图例图的背景透明，这样保存的图片只有图例本身
    plt.tight_layout()

    print(f'Build Legend Chart Done({time.time() - start_time:.1f} s).')

    # 保存图例图
    start_time = time.time()
    print(f'Save Legend Chart...', end='\r')
    legend_figure.savefig(f'./visualize/{dataset_name}_legend.pdf', bbox_inches='tight')
    plt.close(legend_figure)  # 关闭图例 figure
    print(f'Save Legend Chart Done({time.time() - start_time:.1f} s).')


def draw_scatter(x: np.ndarray, y: Union[List[str], np.ndarray], title: str, file_name: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    label_map = get_label_map(title)

    start_time = time.time()
    print(f'Build Chart...', end='\r')

    unique_labels = sorted(set(label_map.values()))
    palette = sns.color_palette("colorblind", len(unique_labels))
    # 使用Seaborn绘制散点图，并按类别上色
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, palette=palette, s=0.5)

    plt.axis('off')
    print(f'Build Chart Done({time.time() - start_time:.1f} s).')

    start_time = time.time()
    print(f'Save Chart...', end='\r')
    # 显示图像或保存成文件
    from os import makedirs
    makedirs('./visualize', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'./visualize/{file_name}.pdf', bbox_inches='tight')
    plt.clf()
    print(f'Save Chart Done({time.time() - start_time:.1f} s).')


def manifold(x, seed: int = 18, umap: bool = False):
    # 降维流量编码到二维, 这样可以直接将数据作为横纵坐标, 直接以散点图的形式显示在平面直角坐标系上
    print(f'Dimensionality Reduction...', end='\r')
    start_time = time.time()
    if umap:
        # UMAP降维
        import umap
        x_manifold = umap.UMAP(n_components=2).fit_transform(x)
    else:
        from sklearnex import patch_sklearn
        patch_sklearn()
        from sklearnex.manifold import TSNE
        x_manifold = TSNE(n_components=2, random_state=seed, n_jobs=-1).fit_transform(x)
    print(f'Dimensionality Reduction Done({time.time() - start_time:.1f} s).')
    return x_manifold


def visualize_flow_encode(x: np.ndarray, y: np.ndarray, file_name: str, title: str = '', seed: int = 18) -> None:
    """
    通过TSNE, 将流量编码降维到二维, 以散点图形式可视化流量编码效果
    注意: 对于大规模数据, TSNE降维和可视化可能是耗时操作

    :param x:           type=np.ndarray, 流编码模型编码后的流量编码
    :param y:           type=np.ndarray, 流量编码对应的流量类型标签
    :param file_name:   type=str, 可视化输出的png文件名
    :param title:       type=str, 散点图名称
    :param seed:        type=int, TSNE降维使用的随机种子, default=18
    """
    x_tsne = manifold(x, seed)
    draw_scatter(x_tsne, y, title, file_name)


def graph_visualize(edge_embeds, dataset, seed: int = 18, stream_aggregate: bool = False, stat_feat: bool = False) -> None:

    dataset_name = dataset.split('_', 1)[-1].upper()

    encode_str = 'F' if stream_aggregate else ''
    stat_str = 'S' if stat_feat else ''

    encode_suffix = f'{encode_str}{stat_str}T'
    title = f'{dataset_name}_{encode_suffix}'

    os.makedirs('./output', exist_ok=True)

    data_indices = np.load(f'./output/DIY_index_65.npy')
    embeds = edge_embeds[data_indices]
    np.save(f'./output/embed/{title}_raw_65.npy', embeds)
    np.save(f'./output/{title}_65.npy', manifold(embeds, seed))
    # umap = (dataset_name == 'DIY' and encode_suffix == 'ST') or (dataset_name.startswith('CIC') and encode_suffix == 'FT')
    np.save(f'./output/{title}_umap_65.npy', manifold(embeds, seed, True))
    return

    visualize_flow_encode(edge_embeds, edge_label, title, title, seed)

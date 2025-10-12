from argparse import ArgumentParser
from os import makedirs
from os.path import exists
from typing import Sequence, Dict, Union

import pandas as pd

num_diy_classes = 19
num_iot_classes = 16

# num_diy_classes = len(load_pickle('./datasets/diy/label_map.pkl'))
# num_iot_classes = len(load_pickle('./datasets/CICIOT2023/label_map.pkl'))

dataset_numClasses_map = {
    "new_UNSW-NB15": 10,
    "UNSW-NB15": 10,
    "TON_IOT": 10,
    "new_CICIDS2017": 7,
    "BON_IOT": 4,
    "darknet": 9,
    "CICDDoS2019": 10,
    "ISCX2012": 4,
    'diy': num_diy_classes,
    # 'diy_pool': num_diy_classes,
    'CICIOT2023': num_iot_classes,
    # 'CICIOT2023_pool': num_iot_classes,
}


def parse_arg():
    p = ArgumentParser()
    p.add_argument('--dataset',
                   help='Experimental dataset.',
                   type=str,
                   default="TON_IOT",
                   choices=list(dataset_numClasses_map.keys()))
    p.add_argument('--binary',
                   help='Perform binary or muticlass task',
                   type=lambda x: x.lower() == 'true',
                   default=False)
    p.add_argument('--epochs',
                   help='train epochs',
                   type=int,
                   default=2)
    p.add_argument('--train_batch_size',
                   help='Train batch size',
                   type=int,
                   default=1024)
    p.add_argument('--eval_batch_size',
                   help='Eval batch size',
                   type=int,
                   default=1024)
    p.add_argument('--dropout',
                   help='dropout',
                   type=float,
                   default=0.2)
    p.add_argument('--lr',
                   help='learning rate',
                   type=float,
                   default=7e-3)
    p.add_argument('--loss',
                   choices=('weighted', 'none', 'focal'),
                   type=str,
                   default='none')
    p.add_argument("--isStreamAgg",
                   help="Whether to use feature aggregation",
                   type=lambda x: x.lower() == 'true',
                   default=True)
    p.add_argument('--visualize',
                   help='Visualize train result',
                   type=lambda x: x.lower() == 'true',
                   default=False)
    p.add_argument('--seed',
                   help='Random seed',
                   type=int,
                   default=3407)
    p.add_argument('--num_layers',
                   help='Num layers of EGAD',
                   type=int,
                   default=2)
    p.add_argument('--num_samples',
                   help='Sample nodes or edges before doing Graph Convolution',
                   type=int,
                   default=200)
    p.add_argument('--save',
                   help='Save PyTorch Model State',
                   type=lambda x: x.lower() == 'true',
                   default=True)
    p.add_argument('--device',
                   help='GPU No.',
                   type=int,
                   default=0)
    p.add_argument('--sparse',
                   help='Use SparseTensor for less GPU memory use',
                   type=lambda x: x.lower() == 'true',
                   default=True)
    p.add_argument('--graph',
                   help='Use GNN or RandomForest Only',
                   type=lambda x: x.lower() == 'true',
                   default=True)
    p.add_argument('--edge_feat',
                   help='Use Edge Features',
                   type=lambda x: x.lower() == 'true',
                   default=True)
    p.add_argument('--aggregate_type',
                   choices=['N2N', 'N2E', 'N2N_N2E'],
                   help='Use Attention or Pooling when edges aggregating',
                   type=str,
                   default='N2N_N2E')
    p.add_argument('--visual',
                   help='Visualize the Edge Feat',
                   type=lambda x: x.lower() == 'true',
                   default=False)
    p.add_argument('--train_size',
                   help='Ratio of Train Data',
                   type=float,
                   default=0.7)
    p.add_argument('--stat_feat',
                   help='Use Both Statistics Features and Stream Aggregate Features',
                   type=lambda x: x.lower() == 'true',
                   default=False)
    p.add_argument('--graph_visual',
                   help='Visual Flow Embeddings after GNN model Training',
                   type=lambda x: x.lower() == 'true',
                   default=False)
    p.add_argument('--flow_attention',
                   type=lambda x: x.lower() == 'true',
                   default=True)
    p.add_argument('--flow_rnn',
                   type=lambda x: x.lower() == 'true',
                   default=True)
    return p.parse_args()


def parse_train_result(train_res_dic: Dict[str, Union[float, int, str]], train_args, ignored_keys: Sequence[str])\
        -> Dict[str, Union[float, int, str]]:
    import time
    train_args.time = int(time.time()-train_args.time)
    train_res_dic.update(vars(train_args))
    for k in ignored_keys:
        if k in train_res_dic:
            del train_res_dic[k]

    train_res = pd.DataFrame(data={k: [v] for k, v in train_res_dic.items()})
    res_save_path = './result/res_new.csv'
    makedirs(res_save_path.rsplit('/', 1)[0], exist_ok=True)
    if exists(res_save_path):
        train_res.to_csv(res_save_path, index=False, header=False, mode='a')
    else:
        train_res.to_csv(res_save_path, index=False, mode='w')

    return train_res_dic

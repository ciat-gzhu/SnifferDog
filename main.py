import torch
from beeprint import pp

from utils.config import dataset_numClasses_map, parse_arg, parse_train_result
from utils.data import load_data_and_process
from utils.train import train

if __name__ == '__main__':
    args = parse_arg()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.eval_batch_size = args.train_batch_size

    args.eval_batch_size = args.train_batch_size = 8192
    task = 'bin' if args.binary else 'mul'
    num_classes = 2 if args.binary else dataset_numClasses_map[args.dataset]
    data = load_data_and_process(args.dataset, args.isStreamAgg, args.stat_feat, args.binary,
                                 args.train_size, device,
                                 need_edge_feat=args.edge_feat, need_graph=args.graph,
                                 flow_attention=args.flow_attention, flow_rnn=args.flow_rnn,
                                 feat_visualize=args.visual, random_seed=args.seed)
    pp(vars(args))
    res = train(args, data, device, num_classes=num_classes, random_seed=args.seed)
    res = parse_train_result(res, args, ignored_keys=[
                                                      'seed', 'visualize',
                                                      'device', 'sparse', 'lr',
                                                      'dropout',
                                                      'train_batch_size',
                                                      'eval_batch_size',
                                                      'save', 'visual', 'graph_visual',
                                                      # 'cm'
                                                      # 'isStreamAgg',
                                                      # 'stat_feat'
                                                      ])

from os.path import exists
from typing import List, Tuple, Optional, Union, Dict, Set, Any

import numpy as np
import torch

from utils.data import load_json_data


# 数据Z-score标准化
def z_score_normalize(data: np.ndarray) -> np.ndarray:
    # 必须保证数组为二维数组, 若不是则将shape=(N)的数组先转为(N, 1)计算, 在返回前再展平
    need_trans = (len(data.shape) == 1)
    if need_trans:
        data = data[:, np.newaxis]
    # 计算每行数据的均值和标准差
    mean = np.nanmean(data, axis=1, keepdims=True)
    std = np.nanstd(data, axis=1, keepdims=True)
    # 将标准差为0的值替换为一个很小的值, 避免除零错误
    std[std == 0] = 1e-6
    # 对每行数据进行Z-score标准化
    normalized_data = np.nan_to_num((data - mean) / std)
    if need_trans:
        normalized_data = normalized_data.ravel()
    return normalized_data


def parse_pcap_to_json(pcap_file_path: str, batch_index: Optional[Union[str, int]] = None,
                       goal_file_name: Optional[str] = None,
                       payload_size: int = 100,
                       num_window_packets: int = 10, skip_exists: bool = False)\
        -> Union[str, None]:
    """
    调用capture工具, 将.pcap或.pcapng文件转换为.json文件
    :param pcap_file_path:      type=str, .pcap或.pcapng文件路径
    :param batch_index:         Optional, type=str, 批次索引, default=None
    :param goal_file_name:      type=str, 生成的目标文件名
    :param payload_size:        type=int, 要生成的json文件中, 每个数据包包含的payload字节数
    :param num_window_packets:  type=int, 每条流最小数据包数
    :return:                    type=str, 返回生成的.json文件路径
    """
    batch_index = '' if batch_index is None else batch_index

    if goal_file_name is None:
        dir_path = pcap_file_path.rsplit('/', 1)[0]
        goal_path = f'{dir_path}/raw_x{batch_index}_du8_p{payload_size}_w{num_window_packets}.json'
    else:
        goal_path = f'{goal_file_name}.json'

    if skip_exists and exists(goal_path):
        print(f'SKIP {goal_path}')
        return

    # --stride=8表示使用int8, -U表示使用无符号整数, 即使用uint8,
    # -tTu4分别表示 使用TCP头、时间戳、UDP头和IPv4头, -Z 0 表示若不存在则填充0
    # -p {payload_size}表示获取Payload中前payload_size字节长度数据
    # -L {num_window_packets }表示一条网络流中最少数据包数为num_window_packets, 否则舍弃
    # -R 100 表示一条网络流中最大数据包数为100, 否则分割
    # -P -W 分别表示要处理的pcap文件路径和输出文件路径
    command = f"./hd-dead -S8 -UCTV -p{payload_size} -L{num_window_packets} -R100 -P{pcap_file_path} -W{goal_path} -J8 --filter=\"(ip or vlan) and (not (net 224.0.0.0/3 or host 255.255.255.255 or host 0.0.0.0))\""

    import subprocess
    print(f'\nExecute command: {command}')
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        return goal_path
    except subprocess.CalledProcessError as e:
        print("命令执行出错:", e)
        exit()
    except Exception as e:
        print("发生了异常:", e)
        exit()


def build_slide_window(flow_data_list: List[np.ndarray],
                       num_window_packets: int = 5,
                       window_stride: int = 1,
                       feature_normalize: bool = True,
                       predict: bool = False)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    构造模型训练需要的滑动窗口对应向量, shape=(num_flows*num_flow_slide, num_window_packets*packet_length)
    对于每个流, 滑动窗口的滑动次数num_flow_slide计算方法如下:
        flow_length = num_flow_packets * packet_length
        num_flow_slide = (flow_length - window_size) // stride + 1
        if (flow_length - window_size) % stride != 0:
            num_flow_slide += 1

    :param flow_data_list:       type=List[np.ndarray], len=num_flows, ndarray.shape=(num_flow_packets, packet_length)
    :param num_window_packets:   type=int, 滑动窗口大小, 即滑动窗口内包的个数
    :param window_stride:        type=int, 滑动窗口滑动步长, todo 目前只实现了stride=1的情况
    :param feature_normalize:    type=bool, 在构造滑动窗口时, 对原始数据是否要进行Z-Score标准化
    :return:
           window_arr:           构造了所有流进入模型的滑动窗口
                                 type=np.ndarray, dtype=np.float32, shape=(num_flows*num_flow_slide, window_size)
           flow_index_arr:       构造了所有流进入模型的滑动窗口对应的流索引, 每行为一个流的滑动窗口的起始索引和终止索引
                                 type=np.ndarray, dtype=np.int32, shape=(num_flows, 2)
    """
    window_index, window_list, flow_index_list = 0, [], []

    for flow_index, flow in enumerate(flow_data_list):
        if len(flow) < num_window_packets:
            continue

        # 对流构造滑动窗口
        window_start_index = window_index
        num_need_packets = num_window_packets -1 if predict else num_window_packets
        for left in range(0, len(flow) - num_window_packets + 1, window_stride):
            window_list.append(flow[left: left + num_need_packets].ravel())
            window_index += 1
        # 如果流长度小于滑动窗口大小, 则忽略该流
        if window_index != window_start_index:
            flow_index_list.append((window_start_index, window_index))
    window_arr = np.asarray(window_list, dtype=np.float32)
    flow_index_arr = np.asarray(flow_index_list, dtype=np.uint32)

    if feature_normalize:
        # Z-score标准化, 使数据满足正态分布, 保持预测效果稳定
        window_arr = z_score_normalize(window_arr)
    return window_arr, flow_index_arr


def merge_flow(predict_flows: torch.Tensor, flow_index_arr: np.ndarray) -> torch.Tensor:
    """
    流数据在预测前被分割为滑动窗口, 并保留了合并所需的流索引,
    该函数将模型预测结果按照流索引进行合并
    :param predict_flows: 流按滑动窗口构造后对应的模型预测结果, shape=(num_flows*num_flow_slide, 128)
    :param flow_index_arr: 流索引, 每行为一个流的滑动窗口的起始索引和终止索引, shape=(num_flows, 2)
    :return: merged_flows: 合并后的流数据, shape=(num_flows, 128)
    """
    merged_flows = torch.zeros((len(flow_index_arr), predict_flows.shape[1]), dtype=predict_flows.dtype)
    for i in range(len(flow_index_arr)):
        merged_flow = predict_flows[flow_index_arr[i, 0]: flow_index_arr[i, 1]]
        merged_flows[i] = merged_flow.sum(dim=0)

        flow_length = (flow_index_arr[:, 1] - flow_index_arr[:, 0] + 1).astype(np.int32)
        flow_length = torch.as_tensor(flow_length, dtype=torch.int32).unsqueeze(1)
        merged_flows = torch.div(merged_flows, flow_length)
    return merged_flows


# 过滤非法IP
def filter_flow(flow_list: List[Dict[str, Any]]):
    def filter_item(flow_data: Dict[str, Any]) -> bool:
        src_ip, dst_ip, _ = flow_data['flowId'].split('_', 2)
        src_ip_part1 = int(src_ip.split('.', 1)[0])
        if src_ip_part1 == 0 or src_ip_part1 == 127 or 224 < src_ip_part1 < 239 or src_ip_part1 == 255:
            return False
        dst_ip_part1 = int(dst_ip.split('.', 1)[0])
        if dst_ip_part1 == 0 or dst_ip_part1 == 127 or 224 < dst_ip_part1 < 239 or dst_ip_part1 == 255:
            return False
        return True
    return list(filter(filter_item, flow_list))


def parse_json_to_flow(json_data_path: str, packet_len: int)\
        -> Tuple[List[str], List[np.ndarray]]:
    """
    读取.json文件, 处理成流信息、流数据和时间戳数组
    :param json_data_path:  type=str, 要读取的.json文件路径
    :param packet_len:      type=int, 要读取的数据包长度, 即不考虑剔除IP、端口和添加时间戳的原始数据包长度
    :return:
           flow_info_list:
                            type=List[Dict[str]], shape=(num_flows), dtype=Dict[str], 每个元素为一个str字典, 记录了每条网络流的信息,
                            内容为{
                                'five_tuple': type=str, 格式为'srcIp_dstIp_srcPort_dstPort_protocol'的五元组
                                'timestamp':  type=float, 网络流时间戳, 即流中第一个数据包的时间戳
                                'offset':     type=int, -1,
                            },
                            当need_flow_info=False时, flow_info_list为None
           new_flow_data_list:
                            type=List[np.ndarray], shape=(num_flows), 每个元素为一个np.ndarray, 记录了每条网络流的所有数据包数据
                            内容为shape=(num_flow_packets, real_packet_length), dtype=np.float32
    """
    # 从.json文件中读取得到格式为List[Dict]的网络流数据对象
    flow_data_list: List[Dict[str, Union[str, List[Dict[str, Union[int, List[int]]]]]]] = load_json_data(json_data_path)
    flow_data_list = filter_flow(flow_data_list)

    flow_info_list, new_flow_data_list, = [], []
    # 遍历List[Dict], 逐条处理网络流数据
    for json_flow_data in flow_data_list:
        protocol = json_flow_data['flowId'].rsplit('_', 1)[1]

        # 读取该网络流中所有数据包数据
        flow_data: List[Dict[str, Union[int, List[int]]]] = json_flow_data['data']
        # 获取该条网络流的时间戳, 也用作后面的相对时间戳计算
        new_flow_data = []
        # 逐数据包处理数据
        for packet_data in flow_data:
            # 根据协议, 进行数据剔除并构造得到新的网络流数组
            new_flow_data.append(
                process_packet_data(
                    np.fromstring(string=packet_data['bitvec'], dtype=np.uint8, count=packet_len, sep=','),
                    protocol
                )
            )
        new_flow_data_list.append(np.asarray(new_flow_data, dtype=np.uint8))
        flow_info_list.append(json_flow_data['flowId'].rsplit('-', 1)[0])
    return flow_info_list, new_flow_data_list


# 根据协议为TCP/UDP, 判断是否从其中剔除IP和端口数据
def process_packet_data(packet_data: np.ndarray, protocol: Optional[str] = None)\
        -> np.ndarray:
    if protocol == 'TCP':
        return np.concatenate([packet_data[:12], packet_data[20:60], packet_data[64:]])
    else:
        return np.concatenate([packet_data[:12], packet_data[20:120], packet_data[124:]])


# 在[start_payload, end_payload]之间寻找可用的数据文件
def get_available_payload(path_prefix: str, file_suffix: str, num_window_packets: int, start_payload: int, end_payload: int = 1500) -> int:
    for payload in range(start_payload, end_payload):
        if exists(f'{path_prefix}_p{payload}_w{num_window_packets}.{file_suffix}'):
            return payload
    return -1

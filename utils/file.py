import pickle
from os import makedirs


def load_pickle(file_path: str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(file_path: str, data):
    file_dir_path = file_path.rsplit('/', 1)[0]
    makedirs(file_dir_path, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

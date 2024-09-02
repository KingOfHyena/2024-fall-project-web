import os
import random
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, data_path, norm: bool = True, cache_path='cache') -> None:
        data = []
        for path in tqdm(data_path):
            if os.path.isdir(path):
                # 如果是目录，遍历目录下的所有文件
                files = os.listdir(path)
            else:
                # 如果是文件路径，将其放入一个列表中
                files = [os.path.basename(path)]
                path = os.path.dirname(path)  # 获取文件的目录路径

            cache_name = f'{os.path.split(path)[-1]}{"_norm" if norm else ""}.cache'
            if cache_path and cache_name in os.listdir(cache_path):
                with open(os.path.join(cache_path, cache_name), 'rb') as f:
                    data += pickle.load(f)
            else:
                path_data = []
                for xls_file in tqdm(files):
                    file_name, ext = os.path.splitext(xls_file)
                    if ext == '.xlsx':
                        with pd.ExcelFile(os.path.join(path, xls_file)) as xls:
                            xls_sheet = xls.parse('力值')
                        max_force = [r['最大力值'] for r in xls_sheet.to_dict('records')]
                        xls_data = xls_sheet.iloc[:, 9:9 + 40 * 26].to_dict('split')['data']
                        path_data += [
                            {
                                'label': int(file_name) - 1,
                                'data': torch.tensor(d, dtype=torch.float32).reshape((1, 26, 40)).mT / f * 1000
                                if norm else torch.tensor(d, dtype=torch.float32).reshape((1, 26, 40)).mT
                            } for d, f in zip(xls_data, max_force)
                        ]
                    elif ext == '.txt':
                        txt_df = pd.read_csv(os.path.join(path, xls_file), header=None)
                        max_force = txt_df.max(1).to_list()
                        txt_data = txt_df.to_dict('split')['data']
                        path_data += [
                            {
                                'label': int(file_name[-1]) - 1,
                                'data': torch.tensor(txt_data[i * 40:(i + 1) * 40], dtype=torch.float32).reshape(
                                    (1, 40, 26)) / max(max_force[i * 40:(i + 1) * 40]) * 1000
                                if norm else torch.tensor(txt_data[i * 40:(i + 1) * 40], dtype=torch.float32).reshape(
                                    (1, 40, 26))
                            } for i in range(len(txt_data) // 40)
                        ]
                if cache_path:
                    with open(os.path.join(cache_path, cache_name), 'wb+') as f:
                        pickle.dump(path_data, f)
                data += path_data

        random.shuffle(data)
        self.data = data

        print('data length', len(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_sample(self, label):
        for d in self.data:
            if d['label'] == int(label) - 1:
                return d['data']

    def select(self):
        return random.choice(self.data)


if __name__ == '__main__':
    d = MyDataset(['data/rxy'])

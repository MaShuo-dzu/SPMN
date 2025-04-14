import csv
import glob
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.data import AgentTrainIter


class CSVTextCosineSimilarityDataset(Dataset):
    def __init__(self, file_paths):
        """
        初始化数据集
        :param file_paths: 包含多个CSV文件路径的列表
        """
        self.data = []

        # 读取每个CSV文件中的数据
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 3:
                        sentence1, sentence2, similarity = row
                        self.data.append((sentence1, sentence2, float(similarity)))

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        :param idx: 索引
        :return: (sentence1, sentence2, similarity)
        """
        return self.data[idx]


class TextDataset(Dataset):
    def __init__(self, file_path: str):
        """
        初始化TextDataset类。

        :param file_path: txt文件的路径（一个）
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            self.sentences = file.readlines()
        # 去除每行末尾的换行符
        self.sentences = [sentence.strip() for sentence in self.sentences]

    def __len__(self):
        """
        返回数据集中句子的数量。
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        根据索引idx返回对应的句子。

        :param idx: 句子的索引
        :return: 索引idx处的句子
        """
        return self.sentences[idx]


class AgentTrainDataset(Dataset):
    def __init__(self, npz_dir: str, dict_file_path: str, memory_threshold: float = 0.2):
        assert os.path.isdir(npz_dir), f"[AgentTrainDataset error] 文件夹不存在：{npz_dir}"
        assert os.path.isfile(dict_file_path), f"[AgentTrainDataset error] 文件不存在：{dict_file_path}"

        # dict
        dict_data = np.load(dict_file_path, allow_pickle=True)
        dict_embeddings = dict_data['embeddings']
        dict_embeddings = torch.cat(dict_embeddings.tolist(), dim=0)

        npz_list = glob.glob(os.path.join(npz_dir, '*.npy'))
        print(f"[AgentTrainDataset] npz 数量：{len(npz_list)}")

        self.scene = []
        count = 0
        for each_npz in tqdm(npz_list, desc="[AgentTrainDataset] loading npz ..."):
            file_data = np.load(each_npz, allow_pickle=True)
            count += file_data.size

            iter_list = []
            for each_iter in file_data:  # NpzData
                memory_number = len(each_iter.similarity)
                linear_sequence = torch.linspace(0, 1, steps=memory_number)

                c_pass = each_iter.similarity > memory_threshold
                similarity: Tensor = each_iter.similarity[c_pass]  # [real_num]

                index: Tensor = each_iter.index[c_pass]  # [real_num]

                if len(index):
                    embeddings = dict_embeddings[index]  # [real_num, data_dim]
                else:
                    embeddings = torch.Tensor([])

                p: Tensor = linear_sequence[c_pass]  # [real_num]

                target = torch.cat((p.unsqueeze(-1), similarity.unsqueeze(-1), embeddings), dim=1)
                train_iter = AgentTrainIter(dict_embeddings[each_iter.sentence].unsqueeze(0), target)
                iter_list.append(train_iter)

            self.scene.append(iter_list)

        print(f"[AgentTrainDataset] 加载数据样本（iter）{count}条")

    def __len__(self):
        return len(self.scene)

    def __getitem__(self, idx):
        return self.scene[idx]


import csv
from torch.utils.data import Dataset


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
        self.sentences = [sentence.strip() for sentence in self.sentences[:500000]]

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


class AgentTrainDataset:
    pass
import csv
import glob
import os
import re

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


class BabiDataset(Dataset):
    def __init__(self, dir_path, task_id, is_train=True, max_story_len=100, max_question_len=20, word2idx=None, build_vocab=True):
        """
        :param dir_path: bAbI 数据集所在目录
        :param task_id: 任务编号 (1-20)
        :param is_train: 是否加载训练集（True）或测试集（False）
        """
        self.task_id = task_id
        self.is_train = is_train
        self.max_story_len = max_story_len
        self.max_question_len = max_question_len
        self.word2idx = word2idx if word2idx else {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab = build_vocab
        self.data = []

        filename = self._find_file(dir_path)
        if filename is None:
            raise FileNotFoundError(f"Task {task_id} file not found in {dir_path}")
        self._parse_file(filename)

    def _find_file(self, dir_path):
        """自动找到对应任务的文件"""
        keyword = f"qa{self.task_id}_"
        postfix = "train" if self.is_train else "test"
        for fname in os.listdir(dir_path):
            if keyword in fname and postfix in fname and fname.endswith(".txt"):
                return os.path.join(dir_path, fname)
        return None

    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def _parse_file(self, file_path):
        with open(file_path, "r") as f:
            story = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx, text = line.split(" ", 1)
                if '\t' in text:
                    question, answer, _ = text.split('\t')
                    q_tokens = self._tokenize(question)
                    s_tokens = [tok for s in story for tok in self._tokenize(s)]

                    if self.build_vocab:
                        for token in s_tokens + q_tokens + [answer]:
                            if token not in self.word2idx:
                                self.word2idx[token] = len(self.word2idx)

                    story_ids = [self.word2idx.get(tok, 1) for tok in s_tokens][-self.max_story_len:]
                    question_ids = [self.word2idx.get(tok, 1) for tok in q_tokens][:self.max_question_len]
                    answer_id = self.word2idx.get(answer, 1)

                    story_ids += [0] * (self.max_story_len - len(story_ids))
                    question_ids += [0] * (self.max_question_len - len(question_ids))

                    self.data.append((story_ids, question_ids, answer_id))
                else:
                    story.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        story, question, answer = self.data[idx]
        return (
            torch.tensor(story, dtype=torch.long),
            torch.tensor(question, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

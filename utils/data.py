import numpy as np
from torch import Tensor


class AgentTrainIter:
    embedding: Tensor
    target: Tensor

    def __init__(self, embedding, target):
        self.embedding = embedding
        self.target = target


class NpzData:
    sentence: int
    similarity: Tensor
    index: Tensor

    def __init__(self, sentence, similarity, index):
        self.sentence = sentence
        self.similarity = similarity
        self.index = index


if __name__ == '__main__':
    # write
    # data_list = []
    # data1 = NpzData("hello", Tensor([[200, 100, 300]]), [])
    # data2 = NpzData("world", Tensor([[100, 150, 250]]), [0.3])
    # data3 = NpzData("!", Tensor([[150, 150, -250]]), [0.3, 0.2])
    #
    # data_list.append(data1)
    # data_list.append(data2)
    # data_list.append(data3)
    #
    # np.save("test.npy", data_list)

    # read
    loaded_array = np.load(r"E:\pythonProject\SPMN\dataset\sentence\100\0.npy", allow_pickle=True)
    for i in loaded_array:
        print(i)

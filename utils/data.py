import numpy as np
from torch import Tensor


class AgentTrainData:
    sentence: str
    embedding: Tensor
    similarity: list

    def __init__(self, sentence, embedding, similarity):
        self.sentence = sentence
        self.embedding = embedding
        self.similarity = similarity


if __name__ == '__main__':
    # write
    data_list = []
    data1 = AgentTrainData("hello", Tensor([[200, 100, 300]]), [])
    # print(data1.embedding.shape)
    data2 = AgentTrainData("world", Tensor([[100, 150, 250]]), [0.3])

    data_list.append(data1)
    data_list.append(data2)

    np.save("test.npy", data_list)

    # read
    loaded_array = np.load("test.npy", allow_pickle=True)
    print(loaded_array[0].embedding)

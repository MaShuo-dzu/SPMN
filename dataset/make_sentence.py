import os

import numpy as np
import torch
from tqdm import tqdm

from utils.csm import compute_cosine_similarity_matrix
from utils.data import NpzData

data = np.load(r"E:\pythonProject\SPMN\dataset\sentences_with_embeddings.npz", allow_pickle=True)

loaded_embeddings = data['embeddings']
sentence_num = loaded_embeddings.size

files = 500
lines = 100
save_dir = r"E:\pythonProject\SPMN\dataset\sentence\{}".format(lines)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

assert lines < sentence_num, "sentence < sentence_num !"

for index in range(files):
    save = []
    np.random.seed(index)
    file_path = os.path.join(save_dir, f"{index}.npy")

    random_samples_index = np.random.choice(sentence_num, size=lines, replace=False)
    random_samples = loaded_embeddings[random_samples_index]
    # random_samples[0].shape  # torch.Size([1, 384])
    save.append(NpzData(random_samples_index[0], torch.Tensor([]), torch.Tensor([])))

    for i in tqdm(range(1, lines), desc=f"{index + 1} / {files}"):
        sentence = random_samples_index[i]
        index = torch.tensor(random_samples_index[:i])
        similarity = compute_cosine_similarity_matrix(
            random_samples[i],  # Tensor
            torch.cat((random_samples[:i].tolist()), dim=0)  # array(Tensor)
        )  # [1, num]

        save.append(NpzData(sentence, similarity.squeeze(0), index))

    np.save(file_path, save)

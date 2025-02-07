import torch
from torch.utils.data import DataLoader

from utils.dataloader import AgentTrainDataset


def custom_collate_fn(batch):
    print(len(batch))

    return "padded_sequences, labels, lengths"


npz_dir = r"./dataset/sentence/100"
dataset = AgentTrainDataset(npz_dir, r"./dataset\sentences_with_embeddings.npz")
train_kwargs = {}
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn, **train_kwargs)

for i in dataloader:
    print(i)

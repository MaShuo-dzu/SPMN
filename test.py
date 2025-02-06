from utils.dataloader import AgentTrainDataset

npz_dir = r"./dataset/sentence/100"
dataset = AgentTrainDataset(npz_dir, r"./dataset\sentences_with_embeddings.npz")

print(len(dataset))

for i in dataset:
    print(i[10].embedding.shape)
    print(i[10].target.shape)
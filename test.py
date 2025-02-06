from utils.dataloader import AgentTrainDataset

npz_dir = r"./dataset/sentence/100"
dataset = AgentTrainDataset(npz_dir, r"./dataset\sentences_with_embeddings.npz")

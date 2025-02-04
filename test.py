import numpy as np

data = np.load(r"E:\pythonProject\SPMN\dataset\sentences_with_embeddings.npz", allow_pickle=True)
loaded_sentences = data['sentences']
loaded_embeddings = data['embeddings']

print(loaded_sentences[:5])
print(loaded_embeddings[:5])

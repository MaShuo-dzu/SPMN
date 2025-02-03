import numpy as np
from tqdm import tqdm

from utils.sentence_embedding import sentence_embedding

# 1. 读取txt文件
txt_file = "text.txt"

# 读取所有句子到列表
with open(txt_file, mode='r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]
    print(len(sentences))

# 2. 加载模型并生成嵌入
embeddings = []
for sentence in tqdm(sentences):
    embeddings.append(sentence_embedding(sentence).cpu())  # 直接得到 numpy 数组

# 3. 保存为npy文件
# 保存（合并为一个文件）
np.savez("sentences_with_embeddings.npz",
         sentences=np.array(sentences),
         embeddings=np.array(embeddings))

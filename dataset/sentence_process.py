import numpy as np
from tqdm import tqdm

from utils.sentence_embedding import sentence_embedding

# 1. ��ȡtxt�ļ�
txt_file = "text.txt"

# ��ȡ���о��ӵ��б�
with open(txt_file, mode='r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]
    print(len(sentences))

# 2. ����ģ�Ͳ�����Ƕ��
embeddings = []
for sentence in tqdm(sentences):
    embeddings.append(sentence_embedding(sentence).cpu())  # ֱ�ӵõ� numpy ����

# 3. ����Ϊnpy�ļ�
# ���棨�ϲ�Ϊһ���ļ���
np.savez("sentences_with_embeddings.npz",
         sentences=np.array(sentences),
         embeddings=np.array(embeddings))

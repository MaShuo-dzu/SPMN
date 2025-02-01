"""
多种词嵌入方式，目的为训练记忆网络，通常此处模型权重要冻结
"""
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 CLIP 文本编码器
tokenizer = AutoTokenizer.from_pretrained(r"G:\ms\SPMN\gte-base-zh")
text_model = AutoModel.from_pretrained(r"G:\ms\SPMN\gte-base-zh").to(device)
# tokenizer = AutoTokenizer.from_pretrained(r"G:\ms\SPMN\bge-small-zh")
# text_model = AutoModel.from_pretrained(r"G:\ms\SPMN\bge-small-zh").to(device)
# tokenizer = CLIPTokenizer.from_pretrained(r"G:\ms\SPMN\clip-vit-base")
# text_model = CLIPTextModel.from_pretrained(r"G:\ms\SPMN\clip-vit-base").to(device)

processor = CLIPProcessor.from_pretrained("G:\ms\SPMN\clip-vit-base")
model = CLIPModel.from_pretrained("G:\ms\SPMN\clip-vit-base").to(device)


def embedding_text_512_77(texts: list, seq_len: int = 77):
    assert 3 <= seq_len <= tokenizer.model_max_length, f"3 <= seq_len <= {tokenizer.model_max_length} !"

    # 编码文本
    inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len).to(device)

    # 获取文本特征
    with torch.no_grad():
        text_features = text_model(**inputs).last_hidden_state

    return text_features  # 输出形状为 (batch_size, seq_len, hidden_dim)


def embedding_img_512(path: str):
    image = Image.open(path)  # 替换为你的图像路径

    # 预处理图像
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 获取图像嵌入向量
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    return image_embeddings  # 输出形状为 (batch_size, hidden_dim)

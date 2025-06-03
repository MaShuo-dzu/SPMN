import argparse
import json

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt  # 新增：用于绘制损失图

from utils.dataloader import BabiDataset


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress = tqdm(dataloader, desc="Train", leave=False)
    for story, question, label in progress:
        story, question, label = story.to(device), question.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(story, question)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(label.cpu())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, p, r, f1


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress = tqdm(dataloader, desc="Val", leave=False)
    with torch.no_grad():
        for story, question, label in progress:
            story, question, label = story.to(device), question.to(device), label.to(device)
            outputs = model(story, question)
            loss = criterion(outputs, label)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

            progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, p, r, f1


def plot_losses(train_losses, val_losses, work_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.join(work_dir, "training_plots"), "loss_curves.png"))
    plt.close()


def plot_val_indicators(accs, recalls, ps, f1s, work_dir):
    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制准确率曲线
    plt.plot(accs, label='Accuracy', marker='o')

    # 绘制召回率曲线
    plt.plot(recalls, label='Recall', marker='s')

    # 绘制精确率曲线
    plt.plot(ps, label='Precision', marker='^')

    # 绘制 F1 分数曲线
    plt.plot(f1s, label='F1 Score', marker='*')

    # 添加图表标题和轴标签
    plt.title(f'Validation Indicators')
    plt.xlabel('Epoch')
    plt.ylabel('Value')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True)
    plt.savefig(os.path.join(os.path.join(work_dir, "training_plots"), "indicators.png"))
    plt.close()


def get_arg():
    parser = argparse.ArgumentParser(description="训练参数。")

    parser.add_argument('--config', type=str, help='参数设置目录')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arg()

    with open(args.config, 'r', encoding='utf - 8') as file:
        # 读取文件内容并转换为字典
        param_dict = json.load(file)

    workdir = param_dict["work_dir"]
    set_seed(param_dict["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据集
    train_dataset = BabiDataset(dir_path=param_dict["babi_dir"], task_id=param_dict["task_id"], is_train=True)
    val_dataset = BabiDataset(dir_path=param_dict["babi_dir"], task_id=param_dict["task_id"], is_train=False,
                              word2idx=train_dataset.word2idx, build_vocab=False)

    train_loader = DataLoader(train_dataset, batch_size=param_dict["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=param_dict["batch_size"], shuffle=False)

    # 2. 初始化模型
    model = YourMemoryNetworkModel(vocab_size=len(train_dataset.word2idx),
                                   story_len=train_dataset.max_story_len,
                                   question_len=train_dataset.max_question_len).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param_dict["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    num_epochs = param_dict["epochs"]

    # 新增：记录每个epoch的指标
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs, val_f1s, val_ps, val_rs = [], [], [], []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        # 训练一个epoch
        train_loss, train_acc, train_p, train_r, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证一个epoch
        val_loss, val_acc, val_p, val_r, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_ps.append(val_p)
        val_rs.append(val_r)

        tqdm.write(f"Epoch {epoch}:")
        tqdm.write(
            f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, P: {train_p:.4f}, R: {train_r:.4f}, F1: {train_f1:.4f}"
        )
        tqdm.write(
            f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, P: {val_p:.4f}, R: {val_r:.4f}, F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            tqdm.write("  Saved best model.")

        scheduler.step()

        # 图像绘制
        plot_losses(train_losses, val_losses, workdir)
        plot_val_indicators(val_accs, val_rs, val_ps, val_f1s, workdir)

# coding=utf-8

import csv
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from spmn.spmn import Spmn
from utils.data import AgentTrainLoaderData
from utils.tools import count_parameters, save_arg, make_workdir
from utils.dataloader import AgentTrainDataset
from utils.loss import AgentTrainLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
torch.autograd.set_detect_anomaly(True)


def custom_collate_fn(batch):
    batchs = []

    for col in zip(*batch):
        embeddings = []
        targets = []
        for j in range(len(batch)):  # batch size
            embedding = col[j].embedding
            target = col[j].target

            embeddings.append(embedding)
            targets.append(target)

        atld = AgentTrainLoaderData(torch.cat(embeddings, dim=0), targets)
        batchs.append(atld)

    return batchs, len(batch)


class AgentTrain(object):
    def __init__(self):

        self.memory_width = 512
        self.memory_deep = 10
        self.input_dim = 384

        self.conf_threshold = 0.5

        self.epochs = 100
        self.batch_size = 2
        self.work_dir = r"AgentTrain-run"
        self.lr = 0.0005

        self.work_dir = make_workdir(self.work_dir)

        if not os.path.exists(os.path.join(self.work_dir, "weights")):
            os.mkdir(os.path.join(self.work_dir, "weights"))

        npz_dir = r"./dataset/sentence/100"

        # 配置GPU
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(123)  # 为当前GPU设置随机种子
        else:
            torch.manual_seed(123)  # 为CPU设置种子用于生成随机数，以使得结果是确定的

        self.device = torch.device("cuda" if use_cuda else "cpu")

        # 初始化数据集
        dataset = AgentTrainDataset(npz_dir, r"./dataset\sentences_with_embeddings.npz")

        train_kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn, **train_kwargs)

        # 选择模型
        self.model = Spmn(memory_width=self.memory_width, memory_deep=self.memory_deep,
                          input_dim=self.input_dim,
                          output_dim=self.input_dim
                          ).to(device=self.device)
        print(self.model)
        self.model_version = self.model.version()
        print("模型参数量/训练参数量： ", count_parameters(self.model))

        # 制作训练集批次
        self.batch_num = len(self.dataloader)

        self.train_batch_num = round(self.batch_num * 0.8)

        # 使用gpu进行多卡训练
        if use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        '''
        构造loss目标函数
        选择优化器
        学习率变化选择
        '''
        # 损失
        self.criterion = AgentTrainLoss()
        # 随机梯度下降
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        # 学习率余弦变化
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.epochs, eta_min=1e-5)

        # 全局参数
        self.train_all_batch_loss = []
        self.val_all_batch_loss = []

        self.train_epochs_loss = []
        self.val_epochs_loss = []

        self.lr_schedular = []
        self.min_val_loss = 999999999999

        save_arg(
            {
                "time": time.strftime("%Y%m%d-%H%M%S"),
                "author": "MaShuo",
                "model-vision": self.model_version,
                "memory_width": self.memory_width,
                "memory_deep": self.memory_deep,
                "input_dim": self.input_dim,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch-size": self.batch_size,
                "optimizer": str(self.optimizer),
                "gpus": torch.cuda.device_count(),
                "data-file": npz_dir,
                "work-dir": self.work_dir,
                "model_params": count_parameters(self.model),
            },
            self.work_dir
        )

        # 训练
        print("开始训练")
        # try:
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
        # except Exception as e:
        #     print(f"训练过程中发生错误：{e}")

        # 清空使用过的gpu缓冲区
        torch.cuda.empty_cache()

        # 画图
        self.plot_losses(x=len(self.train_epochs_loss), x_label="epochs",
                         y=self.train_epochs_loss, type="Train")
        self.plot_losses(x=len(self.val_epochs_loss), x_label="epochs",
                         y=self.val_epochs_loss, type="Val")
        self.plot_losses(x=len(self.train_all_batch_loss), x_label="items",
                         y=self.train_all_batch_loss, type="Train")
        self.plot_losses(x=len(self.val_all_batch_loss), x_label="items",
                         y=self.val_all_batch_loss, type="Val")
        self.plot_losses(x=len(self.lr_schedular), x_label="steps",
                         y=self.lr_schedular, type="lr")

        print("训练完成!")

    def train(self, epoch):
        train_loss = []
        val_loss = []

        # 迭代器
        pbar = tqdm(self.dataloader,
                    desc=f'Epoch {epoch} / {self.epochs}')

        for step, (batchs, batch_num) in enumerate(pbar):
            try:
                self.model.init_M(batch_num)
            except:  # ddp
                self.model.module.init_M(batch_num)

            for train_iter in batchs:
                embedding = train_iter.embedding.to(self.device)  # tensor
                target = train_iter.target  # list

                if step < self.train_batch_num:
                    # train
                    self.model.train()
                    self.optimizer.zero_grad()  # 模型参数梯度清零

                    output = self.model(embedding)
                    loss = self.criterion(output.cpu(), target)

                    loss.backward()

                    train_loss.append(loss.item())
                    self.train_all_batch_loss.append(loss.item())

                    # 调整参数
                    self.optimizer.step()

                    pbar.set_description(
                        f'Train Epoch:{epoch}/{self.epochs} train_loss:{round(loss.item(), 4)}')
                else:
                    # val
                    self.model.eval()

                    with torch.no_grad():
                        output = self.model(embedding)
                        loss = self.criterion(output.cpu(), target)

                    val_loss.append(loss.item())
                    self.val_all_batch_loss.append(loss.item())

                    if loss.item() < self.min_val_loss:
                        self.min_val_loss = loss.item()

                        torch.save(self.model.state_dict(),
                                   os.path.join(self.work_dir, "weights" + f'/best.pth')
                                   )

                    pbar.set_description(
                        f'Val Epoch:{epoch}/{self.epochs} val_loss:{round(loss.item(), 4)}')

        # 调整学习率
        self.lr_schedular.append(self.optimizer.param_groups[0]['lr'])
        self.scheduler.step()

        # 更新全局损失
        self.train_epochs_loss.append(np.mean(train_loss))
        self.val_epochs_loss.append(np.mean(val_loss))

        # 保存模型参数
        torch.save(self.model.state_dict(),
                   os.path.join(self.work_dir, "weights" + f'/Epoch-{epoch}.pth')
                   )

    def plot_losses(self, x: int, x_label: str, y: list, type: str):
        # 绘制训练和验证损失曲线
        _x = range(1, x + 1)  # epoch 范围
        plt.plot(_x, y, label=type + ' Loss')

        plt.xlabel(x_label)
        plt.ylabel(type + 'loss')
        plt.title(type + 'loss --- ' + x_label)
        plt.legend()
        plt.grid(True)

        # 自动保存图像
        plt.savefig(os.path.join(self.work_dir, f"{type + '_' + x_label}_loss.png"), format='png', dpi=300)

        # 显示图像
        plt.show()
        plt.close()  # 关闭图像，释放内存

        with open(os.path.join(self.work_dir, f"{type + '_' + x_label}_loss.csv"), 'w', newline='') as csvfile:
            # 创建一个csv写入器
            writer = csv.writer(csvfile)

            # 写入标题行
            writer.writerow([x_label, type + ' Loss'])

            # 写入索引和损失值
            for index, loss in enumerate(y):
                writer.writerow([index, loss])


if __name__ == "__main__":
    begin = AgentTrain()

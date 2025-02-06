"""
����������Ŀ�����ѵ����ʽ
"""
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
from utils.tools import count_parameters, save_arg
from utils.dataloader import AgentTrainDataset
from utils.loss import AgentTrainLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


class AgentTrain(object):
    def __init__(self):

        self.memory_width = 512
        self.memory_deep = 10
        self.input_dim = 384

        self.conf_threshold = 0.5

        self.epochs = 100
        self.work_dir = r"AgentTrain-run"
        self.lr = 0.0005

        self.shuffle = False

        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)

        if not os.path.exists(os.path.join(self.work_dir, "weights")):
            os.mkdir(os.path.join(self.work_dir, "weights"))

        npz_dir = r"./dataset/sentence/100"

        # ����gpu
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(123)  # Ϊ��ǰGPU�����������
        else:
            torch.manual_seed(123)  # ΪCPU�������������������������ʹ�ý����ȷ����

        self.device = torch.device("cuda" if use_cuda else "cpu")

        '''
        ����DataLoader
        '''
        train_kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        # ��ʼ�����ݼ�
        dataset = AgentTrainDataset(npz_dir, r"./dataset\sentences_with_embeddings.npz")

        # ����ѵ��������֤��
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=self.shuffle, **train_kwargs)

        # ѡ��ģ��
        self.model = Spmn(memory_width=self.memory_width, memory_deep=self.memory_deep,
                          input_dim=self.input_dim,
                          output_dim=self.input_dim
                          ).to(device=self.device)
        print(self.model)
        print("ģ�Ͳ�����/ѵ���������� ", count_parameters(self.model))

        # ����ѵ��������
        self.batch_num = len(self.dataloader)

        self.train_batch_num = round(self.batch_num * 0.8)

        # ʹ��gpu���ж࿨ѵ��
        if use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        '''
        ����lossĿ�꺯��
        ѡ���Ż���
        ѧϰ�ʱ仯ѡ��
        '''
        # ��������ʧ
        self.criterion = AgentTrainLoss()
        # ����ݶ��½�
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        # ѧϰ�����ұ仯
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.epochs, eta_min=1e-5)

        # ȫ�ֲ���
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
                "model-vision": self.model.version(),
                "memory_width": self.memory_width,
                "memory_deep": self.memory_deep,
                "input_dim": self.input_dim,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch-size": 1,
                "optimizer": str(self.optimizer),
                "gpus": torch.cuda.device_count(),
                "data-file": npz_dir,
                "work-dir": self.work_dir,
                "model_params": count_parameters(self.model),
                "other-arg":
                    {
                        "shuffle": self.shuffle
                    }
            },
            self.work_dir
        )

        # ѵ��
        print("��ʼѵ��")
        try:
            for epoch in range(1, self.epochs + 1):
                self.train(epoch)
        except Exception as e:
            print(f"ѵ�������з�������{e}")
        finally:
            # ���ʹ�ù���gpu������
            torch.cuda.empty_cache()

            # ��ͼ
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

        print("ѵ�����!")

    def train(self, epoch):
        train_loss = []
        val_loss = []

        # ������
        pbar = tqdm(self.dataloader,
                    desc=f'Epoch {epoch} / {self.epochs}')

        for step, scene in enumerate(pbar):
            print(scene.shape)
            self.model.module.reset_M()

            for train_iter in scene:
                embedding = train_iter.embedding
                target = train_iter.target

                if step < self.train_batch_num:
                    # train
                    self.model.train()
                    self.optimizer.zero_grad()  # ģ�Ͳ����ݶ�����

                    output = self.model(embedding)
                    loss = self.criterion(output, target)

                    loss.backward()

                    train_loss.append(loss.item())
                    self.train_all_batch_loss.append(loss.item())

                    # ��������
                    self.optimizer.step()

                    pbar.set_description(
                        f'Train Epoch:{epoch}/{self.epochs} train_loss:{round(loss.item(), 4)}')
                else:
                    # val
                    self.model.eval()

                    with torch.no_grad():
                        output = self.model(embedding)
                        loss = self.criterion(output, target)

                    val_loss.append(loss.item())
                    self.val_all_batch_loss.append(loss.item())

                    if loss.item() < self.min_val_loss:
                        self.min_val_loss = loss.item()

                        torch.save(self.model.state_dict(),
                                   os.path.join(self.work_dir, "weights" + f'/best.pth')
                                   )

                    pbar.set_description(
                        f'Val Epoch:{epoch}/{self.epochs} val_loss:{round(loss.item(), 4)}')

        # ����ѧϰ��
        self.lr_schedular.append(self.optimizer.param_groups[0]['lr'])
        self.scheduler.step()

        # ����ȫ����ʧ
        self.train_epochs_loss.append(np.mean(train_loss))
        self.val_epochs_loss.append(np.mean(val_loss))

        # ����ģ�Ͳ���
        torch.save(self.model.state_dict(),
                   os.path.join(self.work_dir, "weights" + f'/Epoch-{epoch}.pth')
                   )

    def plot_losses(self, x: int, x_label: str, y: list, type: str):
        # ����ѵ������֤��ʧ����
        _x = range(1, x + 1)  # epoch ��Χ
        plt.plot(_x, y, label=type + ' Loss')

        plt.xlabel(x_label)
        plt.ylabel(type + 'loss')
        plt.title(type + 'loss --- ' + x_label)
        plt.legend()
        plt.grid(True)

        # �Զ�����ͼ��
        plt.savefig(os.path.join(self.work_dir, f"{type + '_' + x_label}_loss.png"), format='png', dpi=300)

        # ��ʾͼ��
        plt.show()
        plt.close()  # �ر�ͼ���ͷ��ڴ�

        with open(os.path.join(self.work_dir, f"{type + '_' + x_label}_loss.csv"), 'w', newline='') as csvfile:
            # ����һ��csvд����
            writer = csv.writer(csvfile)

            # д�������
            writer.writerow([x_label, type + ' Loss'])

            # д����������ʧֵ
            for index, loss in enumerate(y):
                writer.writerow([index, loss])


if __name__ == "__main__":
    begin = AgentTrain()

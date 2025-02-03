from torch import nn


class Recall(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int, recall_num: int = 500, output_dim: int = 2048):
        super(Recall, self).__init__()

        self.memory_width = memory_width
        self.in_channel = memory_deep
        self.out_channel = recall_num

        self.output_dim = output_dim + 2

        # memory_width // 2
        self.block_1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel * 4, 1),
            nn.BatchNorm2d(self.in_channel * 4),
            nn.ReLU(inplace=True)
        )

        # memory_width // 4
        self.block_2 = nn.Sequential(
            nn.Conv2d(self.in_channel * 4, self.in_channel * 8, 2, 2),
            nn.BatchNorm2d(self.in_channel * 8),
            nn.ReLU(inplace=True)
        )

        # memory_width // 8
        self.block_3 = nn.Sequential(
            nn.Conv2d(self.in_channel * 8, self.in_channel * 16, 2, 2),
            nn.BatchNorm2d(self.in_channel * 16),
            nn.ReLU(inplace=True)
        )

        # memory_width // 8
        self.block_4 = nn.Sequential(
            nn.Conv2d(self.in_channel * 16, self.out_channel, 2, 2),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Linear((memory_width // 8) ** 2, self.output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        bs = x.shape[0]

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)

        x = x.view(bs, self.out_channel, -1)

        x = self.head(x)

        return x

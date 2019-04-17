import torch
import torch.nn as nn
import torch.nn.functional as F

N_FILTERS = 32
KERNEL_SIZE = (3, 3)
OUTPUT_DIM = 3  # Number of classes considered
N_CONV_LAYERS = 4


class Learner(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=N_FILTERS, kernel_size=KERNEL_SIZE)
        self.conv_1 = nn.Conv2d(in_channels=N_FILTERS, out_channels=N_FILTERS, kernel_size=KERNEL_SIZE)
        self.conv_2 = nn.Conv2d(in_channels=N_FILTERS, out_channels=N_FILTERS, kernel_size=KERNEL_SIZE)
        self.conv_3 = nn.Conv2d(in_channels=N_FILTERS, out_channels=N_FILTERS, kernel_size=KERNEL_SIZE)

        self.batch_norm = nn.BatchNorm2d(N_FILTERS)

        self.fc = nn.Linear(N_CONV_LAYERS * N_FILTERS, OUTPUT_DIM)

    def forward(self, x):
        out = self.conv_0(x)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        print("Conv0, out", out.shape)

        out = self.conv_1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        print("Conv1, out", out.shape)

        out = self.conv_2(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        print("Conv2, out", out.shape)

        out = self.conv_3(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        print("Conv3, out", out.shape)

        return F.softmax(out)

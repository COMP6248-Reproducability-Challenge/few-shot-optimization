import torch.nn as nn
import torch.nn.functional as F


class CNNLearner(nn.Module):
    def __init__(self, n_filters, kernel_size, output_dim, bn_momentum):
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channels=3, out_channels=n_filters, kernel_size=kernel_size, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1)

        self.batch_norm = nn.BatchNorm2d(n_filters, momentum=bn_momentum)

        self.fc = nn.Linear(4 * n_filters, output_dim)

    def forward(self, x):
        out = self.conv_0(x)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        out = self.conv_1(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        out = self.conv_2(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        out = self.conv_3(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        out = self.fc(out)

        return F.softmax(out)

    def reset_batch_norm(self):
        raise NotImplemented

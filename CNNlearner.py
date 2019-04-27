import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLearner(nn.Module):
    def __init__(self, image_size, n_filters, kernel_size, output_dim, bn_momentum):
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channels=3, out_channels=n_filters, kernel_size=kernel_size, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1)

        self.batch_norm = nn.BatchNorm2d(n_filters, momentum=bn_momentum)

        fc_in = image_size // 2**4
        self.fc = nn.Linear(fc_in * fc_in * n_filters, output_dim)

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

        out = self.fc(out.flatten(1))

        return F.softmax(out, dim=1)

    def reset_batch_norm(self):
        self.batch_norm.reset_running_stats()

    def get_flat_params(self):
        """
        :return: flattened tensor containing all parameter weights of the learner
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def replace_flat_params(self, flat_params):
        """
        Copies the weight value of the input params to the parameters of the learner
        :param flat_params: flattened tensor of len(n_learner_params)
        """
        current_position = 0
        for param in self.parameters():
            num_weights = len(param.flatten())
            corresponding_weights = flat_params[current_position: current_position + num_weights].view_as(param)
            param.data.copy_(corresponding_weights)
            current_position += num_weights

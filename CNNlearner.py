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

        self.batch_norm_0 = nn.BatchNorm2d(n_filters, eps=1e-03, momentum=bn_momentum)
        self.batch_norm_1 = nn.BatchNorm2d(n_filters, eps=1e-03, momentum=bn_momentum)
        self.batch_norm_2 = nn.BatchNorm2d(n_filters, eps=1e-03, momentum=bn_momentum)
        self.batch_norm_3 = nn.BatchNorm2d(n_filters, eps=1e-03, momentum=bn_momentum)

        # Used for easy access to the layers
        self.bn_list = [self.batch_norm_0, self.batch_norm_1, self.batch_norm_2, self.batch_norm_3]

        fc_in = image_size // 2**4
        self.fc = nn.Linear(fc_in * fc_in * n_filters, output_dim)

    def forward(self, x):
        out = self.conv_0(x)
        out = self.batch_norm_0(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        out = self.conv_1(out)
        out = self.batch_norm_1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        out = self.conv_3(out)
        out = self.batch_norm_3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, (2, 2))

        out = self.fc(out.flatten(1))

        return out

    def reset_batch_norm(self):
        self.batch_norm_0.reset_running_stats()
        self.batch_norm_1.reset_running_stats()
        self.batch_norm_2.reset_running_stats()
        self.batch_norm_3.reset_running_stats()

    def get_flat_params(self):
        """
        :return: flattened tensor containing all parameter weights of the learner
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def get_bn_stats(self):
        """
        :return: Tuple containing the batch normalisation layers running stats.
        """
        bn_means, bn_vars = [], []
        for layer in self.bn_list:
            bn_means.append(layer.running_mean)
            bn_vars.append(layer.running_var)
        return bn_means, bn_vars

    def replace_flat_params(self, flat_params, bn_stats=None):
        """
        Copies the weight value of the input params to the parameters of the learner.
        Copies the batch normalisation layers running stats
        :param flat_params: flattened tensor of len(n_learner_params)
        :param bn_means: list of tensors of len(bn_list)
        :param bn_vars: list of tensors len(bn_list)
        """
        current_position = 0
        for param in self.parameters():
            num_weights = len(param.flatten())
            corresponding_weights = flat_params[current_position: current_position + num_weights].view_as(param)
            param.data.copy_(corresponding_weights)
            current_position += num_weights

        # Copy the batch norm running means and variances if needed
        if bn_stats is not None:
            for idx, layer in enumerate(self.bn_list):
                layer.running_mean = bn_stats[0][idx]
                layer.running_var = bn_stats[1][idx]

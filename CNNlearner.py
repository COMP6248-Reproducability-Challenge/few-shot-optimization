import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLearner(nn.Module):
    def __init__(self, image_size, n_filters, kernel_size, output_dim, bn_momentum, in_channels=3):
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, padding=1)
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
        out = self.conv_0(x.float())
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

    def copy_flat_params(self, flat_params):
        """
        COPIES the weight value of the input params to the parameters of the learner.
        :param flat_params: flattened tensor of len(n_learner_params)
        """
        current_position = 0
        for param in self.parameters():
            num_weights = len(param.flatten())
            corresponding_weights = flat_params[current_position: current_position + num_weights].view_as(param)
            param.data.copy_(corresponding_weights)
            current_position += num_weights

    def clone_flat_params(self, flat_params, bn_stats):
        """
        CLONES the parameters so that computation backpropagation graphs ARE connected.
        :param flat_params: flattened tensor of len(n_learner_params)
        :param bn_stats: batch normalization layer's running stats
        """
        current_position = 0
        for current_module in list(self.modules())[1:]:
            weight_len = len(current_module.weight.flatten())
            bias_len = len(current_module.bias)
            new_w = flat_params[current_position: current_position + weight_len].view_as(current_module.weight).clone()
            new_b = flat_params[current_position: current_position + bias_len].view_as(current_module.bias).clone()
            current_position += weight_len
            current_position += bias_len
            # In order to connect the computation graphs we need to access the
            # protected members of the object to retain the backprop flow.
            current_module._parameters["weight"] = new_w
            current_module._parameters["bias"] = new_b

        # Copy the batch norm running means and variances
        for idx, layer in enumerate(self.bn_list):
            layer.running_mean = bn_stats[0][idx]
            layer.running_var = bn_stats[1][idx]

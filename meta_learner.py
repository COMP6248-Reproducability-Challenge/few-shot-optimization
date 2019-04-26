import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaLearnerLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_learner_params):
        """
        Custom Meta Learner LSTM Cell

        :param input_size: The size of the inputs to this cell. In this case, the size of the outputs of the previous LSTM layer
        :param hidden_size: Should be 1
        :param n_learner_params:
        """
        super(MetaLearnerLSTMCell, self).__init__()
        # Weight matrices for the forget and update gates
        self.W_forget = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.W_update = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))

        # Memory Cell containing the memorized parameters of the leaner
        self.memory_cell = nn.Parameter(torch.Tensor(n_learner_params, 1))

        # Bias terms for the forget and update gates (scalar)
        self.b_forget = nn.Parameter(torch.Tensor(1, hidden_size))
        self.b_update = nn.Parameter(torch.Tensor(1, hidden_size))

        self.init_params()

    def init_params(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        nn.init.uniform_(self.b_forget, 4, 6)
        nn.init.uniform_(self.b_update, -5, -4)

    def forward(self, inputs, h_t=None):
        """
        :param inputs: [coordinates, grad]
                coordinates: outputs from the previous LSTM [n_learner_params, input_size]
                grad: gradients from the Learner
        :param h_t: Previous state gate values [p_forget, p_update, p_memory_cell]
                p_forget: [n_learner_params, 1]
                p_update: [n_learner_params, 1]
                p_memory_cell: [n_learner_params, 1]
        """
        coordinates, grad = inputs
        batch_size, _ = coordinates.size()

        if h_t is None:
            # First sequence to be processed
            hidden_size = self.W_forget.size()[1]
            p_forget = torch.zeros((batch_size, hidden_size)).to(self.W_forget.device)
            p_update = torch.zeros((batch_size, hidden_size)).to(self.W_update.device)
            p_memory_cell = self.memory_cell
            h_t = [p_forget, p_update, p_memory_cell]

        p_forget, p_update, p_memory_cell = h_t

        # Expand the biases to be used below
        b_forget = self.b_forget.expand_as(p_forget)
        b_update = self.bI.expand_as(p_update)

        # Equations from the paper
        # forget_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
        # update_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
        forget_t = torch.mm(torch.cat((coordinates, p_memory_cell, p_forget), 1), self.W_forget) + b_forget
        update_t = torch.mm(torch.cat((coordinates, p_memory_cell, p_update), 1), self.WI) + b_update

        # next cell update
        new_c = torch.sigmoid(forget_t).mul(p_memory_cell) - torch.sigmoid(update_t).mul(grad)

        return new_c, [forget_t, update_t, new_c]


class MetaLearner(nn.Module):
    def __init__(self, input_size, hidden_size, n_learner_params):
        """
        :param input_size: int, input size for the first lstm layer
        :param hidden_size: int, output size for the first lstm layer
        :param n_learner_params: int, number of learner parameters
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.custom_lstm = MetaLearnerLSTMCell(input_size, hidden_size, n_learner_params)

    def forward(self, inputs, hidden_state=None):
        """
        :param inputs: From the paper [learner_loss, learner_grad_prep, learner_grad]
                    learner_loss: tensor of shape [1, 2], containing the learner's loss value
                    learner_grad_prep: tensor of shape [n_learner_params, 2], processed learner gradients
                    learner_grad: tensor of shape [n_learner_params], vanilla learner gradients
        :param hidden_state: previous state gate values for both cells. Gets updated by MetaLSTMCell
                    [(lstm_hidden_state, lstm_cell_state), [metalstm_forget_gate, metalstm_update_gate, metalstm_cell_state]]
        """
        learner_loss, learner_grad_prep, learner_grad = inputs
        # Expand the loss to be of size [n_learner_params]
        learner_loss = learner_loss.expand_as(learner_grad_prep)
        inputs = torch.cat((learner_loss, learner_grad_prep), 1)

        # Initialise hidden_state for t=0
        if hidden_state is None:
            hidden_state = [None, None]

        lstm_out, lstm_cx = self.lstm(inputs, hidden_state[0])
        metalstm_out, metalstm_hs = self.custom_lstm([lstm_out, learner_grad], hidden_state[1])

        return metalstm_out.squeeze(), [(lstm_out, lstm_cx), metalstm_hs]

import copy
import math
import torch
import CNNlearner
import data_loader
import numpy as np
import meta_learner
from sklearn.metrics import accuracy_score

TRAIN_PATH = "data/train"
TEST_PATH = "data/test"
VAL_PATH = "data/val"

CUDA_NUM = 0

ITERATIONS = 10000
EVALS = 15              # items used to test acc and loss
CLASSES = 5             # number of classes we differentiate
SHOTS = 5               # items used to train with for each class
EVAL_POINT = 3          # when to assess performance on validation set

# Learner Network parameters
FILTERS = 32
KERNEL_SIZE = 3
OUTPUT_DIM = 5
BN_MOMENTUM = 0.95
CROPPED_IMAGE_SIZE = 128

# Meta Learner Network parameters
INPUT_SIZE = 4
HIDDEN_SIZE = 20

# More parameters
LEARNING_RATE = 0.001
GRADIENT_CLIPPING = 0.25
LEARNER_TRAINING_EPOCHS = 8
BATCH_SIZE = 25

train_acc_hist = []
val_acc_hist = []


def accuracy(predictions, truth):
    """
    :param predictions: torch tensor with output from classifier
    :param truth: numpy array of true labels
    :return: accuracy score
    """
    predictions = predictions.detach().cpu().numpy()
    return accuracy_score(y_true=truth, y_pred=predictions)


# preprocess parameters elementwise according to paper
def preprocess_parameters(parameters):
    p = 10

    pgrad1 = []
    pgrad2 = []
    for x in range(parameters.size()[0]):
        if (torch.abs(parameters[x]) >= math.exp(-p)):
            pgrad1.append(math.log(torch.abs(parameters[x])) / p)
            pgrad2.append(torch.sign(parameters[x]))
        else:
            pgrad1.append(-1)
            pgrad2.append(math.exp(p) * parameters[x])

    # list to tensors
    pgrad1 = torch.FloatTensor(pgrad1)
    pgrad2 = torch.FloatTensor(pgrad2)

    return torch.stack((pgrad1, pgrad2), 1)


# train the learner using the cell state from the meta learner
def train_learner(learner, metalearner, train_inputs, train_labels):
    memory_cell = metalearner.custom_lstm.memory_cell.data
    hidden_states = [None]
    for epoch in range(LEARNER_TRAINING_EPOCHS):
        for i in range(0, len(train_inputs), BATCH_SIZE):
            x = train_inputs[i: i + BATCH_SIZE]
            y = train_labels[i: i + BATCH_SIZE]

            # Give the learner the updated params
            learner.replace_flat_params(memory_cell)
            output = learner(x)
            target = torch.LongTensor(y)

            # Compute loss and accuracy
            celoss = torch.nn.CrossEntropyLoss()
            loss = celoss(output, target)

            # interpret softmax as set of label predictions
            _, predictions = torch.max(output[:], 1)
            acc = accuracy(predictions, target)

            # Compute gradients
            learner.zero_grad()
            loss.backward()

            grad = torch.cat([p.grad.data.view(-1) / BATCH_SIZE for p in learner.parameters()], 0)

            # preprocess the gradients of the learner to handle varying magnitudes
            pgrad = preprocess_parameters(grad)
            ploss = preprocess_parameters(loss.data.unsqueeze(0))

            metalearner_input = [ploss, pgrad, grad.unsqueeze(1)]
            cI, new_h = metalearner(metalearner_input, hidden_states[-1])
            hidden_states.append(new_h)

    return memory_cell


def meta_test(val_dataset, learner, learner_wo_grad, metalearner):
    # Get the data to train and test the model
    x, y = val_dataset.getitem()
    x = x.reshape((SHOTS, SHOTS + EVALS) + x.shape[1:])
    y = y.reshape((SHOTS, SHOTS + EVALS))

    train_x = x[:, :SHOTS].reshape((SHOTS * CLASSES,) + x.shape[2:])
    train_y = y[:, :SHOTS].flatten()

    test_x = x[:, SHOTS:].reshape((EVALS * CLASSES,) + x.shape[2:])
    test_y = y[:, SHOTS:].flatten()

    # Reset the batch norm layers
    learner.reset_batch_norm()
    learner_wo_grad.reset_batch_norm()

    # Set the Laerner with gradients to train mode and the learner without gradients to eval
    learner.train()
    learner_wo_grad.eval()

    # Get and copy the updated parameters for the learner
    cell_state = train_learner(learner, metalearner, train_x, train_y)
    learner_wo_grad.replace_flat_params(cell_state)

    # Get validation set predictions and return accuracy
    output = learner_wo_grad(test_x)
    preds = torch.max(output[:], 1)[1]

    return accuracy(preds, truth=test_y)


def main():
    # Get the data
    train_dataset = data_loader.MetaDataset(TRAIN_PATH, SHOTS, EVALS, CLASSES)
    val_dataset = data_loader.MetaDataset(VAL_PATH, SHOTS, EVALS, CLASSES)

    # Use gpu when possible
    device = torch.device('cuda:' + str(CUDA_NUM) if torch.cuda.is_available() else 'cpu')

    # Create the models
    learner = CNNlearner.CNNLearner(CROPPED_IMAGE_SIZE, FILTERS, KERNEL_SIZE, OUTPUT_DIM, BN_MOMENTUM).to(device)
    # Learner without gradient history
    grad_free_learner = copy.deepcopy(learner)

    metalearner = meta_learner.MetaLearner(INPUT_SIZE, HIDDEN_SIZE, learner.get_flat_params().size(0)).to(device)
    metalearner.init_memory_cell(learner.get_flat_params())

    optimiser = torch.optim.Adam(metalearner.parameters(), lr=LEARNING_RATE)
    learner_loss_function = torch.nn.CrossEntropyLoss()

    # Training Loop
    for it in range(ITERATIONS):
        x, y = train_dataset.getitem()
        x = x.reshape((SHOTS, SHOTS + EVALS) + x.shape[1:])
        y = y.reshape((SHOTS, SHOTS + EVALS))

        train_x = x[:, :SHOTS].reshape((SHOTS * CLASSES,) + x.shape[2:])
        train_y = y[:, :SHOTS].flatten()

        test_x = x[:, SHOTS:].reshape((EVALS * CLASSES,) + x.shape[2:])
        test_y = y[:, SHOTS:].flatten()

        # Train learner with metalearner
        learner.reset_batch_norm()
        grad_free_learner.reset_batch_norm()
        learner.train()
        grad_free_learner.train()
        new_cell_state = train_learner(learner, metalearner, train_x, train_y)

        # new cell state contains our parameters in a 1-d array
        grad_free_learner.replace_flat_params(new_cell_state)

        # grad_free_learner.transfer_params(learner, new_cell_state)
        output = grad_free_learner(test_x)
        predictions = torch.max(output[:], 1)[1]
        loss = learner_loss_function(output, torch.LongTensor(test_y))
        train_acc = accuracy(predictions, test_y)
        train_acc_hist.append(train_acc)
        print("Iteration {} | Training Accuracy {:.4f}".format(it, train_acc))

        optimiser.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(metalearner.parameters(), GRADIENT_CLIPPING)
        optimiser.step()

        # Meta-validation
        if it % EVAL_POINT == 0:
            val_acc = meta_test(val_dataset, learner, grad_free_learner, metalearner)
            print("Iteration {} | Training Accuracy {:.4f} | Validation Accuracy {:.4f}".format(it, train_acc, val_acc))
            val_acc_hist.append(val_acc)

            print("Avg. Train Accuracy: " + str(np.mean(np.array(train_acc_hist))))
            print("Avg. Validation Accuracy: " + str(np.mean(np.array(val_acc_hist))))


if __name__ == "__main__":
    main()

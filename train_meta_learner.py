import copy
import torch
import learner
import data_loader
import meta_learner

iterations = 5
evals = 15  # items used to test acc and loss
classes = 5  # number of classes we differentiate
shots = 5  # items used to train with for each class
train_path = "data/train"
test_path = "data/test"
val_path = "data/val"

# Learner Network parameters
FILTERS = 32
KERNEL_SIZE = 3
OUTPUT_DIM = 5
BN_MOMENTUM = 0.2

# Meta Learner Network parameters
INPUT_SIZE = 64
HIDDEN_SIZE = 256

# More parameters
LEARNING_RATE = 0.01
GRADIENT_CLIPPING = 0.025


def accuracy(predictions, truth):
    raise NotImplementedError


def train_learner(learner_w_grad, metalearner, train_inputs, train_labels):
    raise NotImplementedError


def meta_test(iteration, val_dataset, learner_w_grad, learner_wo_grad, metalearner):
    raise NotImplementedError


def main():
    # Get the data
    train_dataset = data_loader.MetaDataset(train_path, shots, evals, classes)
    val_dataset = data_loader.MetaDataset(val_path, shots, evals, classes)

    # Use gpu when possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create the models
    learner_w_grad = learner.Learner(FILTERS, KERNEL_SIZE, OUTPUT_DIM, BN_MOMENTUM).to(device)
    # Learner without gradient history
    learner_wo_grad = copy.deepcopy(learner_w_grad)

    metalearner = meta_learner.MetaLearner(INPUT_SIZE, HIDDEN_SIZE, learner_w_grad.get_flat_params().size(0)).to(device)
    metalearner.init_memory_cell(learner_w_grad.get_flat_params())

    optim = torch.optim.Adam(metalearner.parameters(), lr=LEARNING_RATE)
    learner_loss_function = torch.nn.CrossEntropyLoss()

    # Training Loop
    for it in range(iterations):
        x, y = train_dataset.getitem()
        x = x.reshape((shots, shots + evals) + x.shape[1:])
        y = y.reshape((shots, shots + evals))

        train_x = x[:, :shots].reshape((shots * classes, ) + x.shape[2:])
        train_y = y[:, :shots].flatten()

        test_x = x[:, shots:].reshape((evals * classes, ) + x.shape[2:])
        test_y = y[:, shots:].flatten()

        # Train learner with metalearner
        learner_w_grad.reset_batch_norm()
        learner_wo_grad.reset_batch_norm()
        learner_w_grad.train()
        learner_wo_grad.train()
        new_cell_state = train_learner(learner_w_grad, metalearner, train_x, train_y)

        # Train meta-learner
        learner_wo_grad.transfer_params(learner_w_grad, new_cell_state)
        output = learner_wo_grad(test_x)
        loss = learner_loss_function(output, test_y)
        acc = accuracy(output, test_y)
        print("Iteration {}, Training Accuracy {}".format(it, acc))

        optim.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(metalearner.parameters(), GRADIENT_CLIPPING)
        optim.step()

        # Meta-validation
        if it != 0:
            acc = meta_test(it, val_dataset, learner_w_grad, learner_wo_grad, metalearner)
            print("Validation accuracy", acc)


main()

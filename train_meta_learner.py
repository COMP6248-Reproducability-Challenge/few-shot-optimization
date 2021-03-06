import os
import time
import copy
import torch
import argparse
import CNNlearner
import data_loader
import meta_learner
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-its', nargs='?', default=10000, type=int, help='Number of training iterations.')
parser.add_argument('-cuda', nargs='?', default=0, type=int, help='Cuda number.')
parser.add_argument('-val', nargs='?', default=1000, type=int, help='When to assess performance on validation set.')
parser.add_argument('-test', default=False, action='store_true')
parser.add_argument('-data', nargs='?', default='MIN', type=str,
                    help='Which dataset to work on: MIN (default: miniImagenet) or MNIST')
parser.add_argument('-classes', nargs='?', default=5, type=int,
                    help='Number of classes, dictates output dims for classifier')
parser.add_argument('-data_root', nargs='?', default='data/', type=str,
                    help='Root for directory whether miniImagenet or other.')
parser.add_argument('-save_state', type=str,
                    help='Name of file to save Metalearner parameters into.')
parser.add_argument('-load_state', type=str,
                    help='Name of file to load Metalearner parameters from.')
parser.add_argument('-channel', nargs='?', default='rgb', type=str, help='rgb or gscale')
parser.add_argument('-shots', nargs='?', default=5, type=int, help='number of shots to learn on')

args = parser.parse_args()

if args.data == 'MIN':
    TRAIN_PATH = os.path.join(args.data_root, 'train')
    TEST_PATH = os.path.join(args.data_root, 'test')
    VAL_PATH = os.path.join(args.data_root, 'val')

if args.data == 'MNIST':
    TRAIN_PATH = TEST_PATH = VAL_PATH = args.data_root  # co-located

CUDA_NUM = args.cuda
ITERATIONS = args.its
EVAL_POINT = args.val

EVALS = 15  # items used to test acc and loss
CLASSES = args.classes  # number of classes we differentiate
SHOTS = args.shots  # items used to train with for each class


# Learner Network parameters
FILTERS = 32
KERNEL_SIZE = 3
INPUT_DIM = 3 if args.channel == 'rgb' else 1
OUTPUT_DIM = args.classes
BN_MOMENTUM = 0.95
CROPPED_IMAGE_SIZE = 84

# Meta Learner Network parameters
INPUT_SIZE = 4
HIDDEN_SIZE = 20

# More parameters
LEARNING_RATE = 0.001
GRADIENT_CLIPPING = 0.25
LEARNER_TRAINING_EPOCHS = 8
BATCH_SIZE = 25

VAL_ITERATIONS = 100

train_acc_hist = []
val_acc_hist = []

# Use gpu when possible
device = torch.device('cuda:' + str(CUDA_NUM) if torch.cuda.is_available() else 'cpu')


def accuracy(predictions, truth):
    """
    :param predictions: torch tensor with output from classifier
    :param truth: numpy array of true labels
    :return: accuracy score
    """
    truth = truth.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    return accuracy_score(y_true=truth, y_pred=predictions)


# preprocess parameters elementwise according to paper
def preprocess_parameters(parameters):
    p = 10
    indicator = (parameters.abs() >= np.exp(-p)).to(torch.float32)

    x_proc1 = indicator * torch.log(parameters.abs() + 1e-8) / p + (1 - indicator) * -1

    x_proc2 = indicator * torch.sign(parameters) + (1 - indicator) * np.exp(p) * parameters
    return torch.stack((x_proc1, x_proc2), 1)


# train the learner using the cell state from the meta learner
def train_learner(learner, metalearner, train_inputs, train_labels):
    memory_cell = metalearner.custom_lstm.memory_cell.data
    hidden_states = [None]
    for epoch in range(LEARNER_TRAINING_EPOCHS):
        for i in range(0, len(train_inputs), BATCH_SIZE):
            x = train_inputs[i: i + BATCH_SIZE]
            y = train_labels[i: i + BATCH_SIZE]

            # Give the learner the updated params
            learner.copy_flat_params(memory_cell)
            output = learner(x)
            target = torch.LongTensor(y).to(device)

            # Compute loss and accuracy
            celoss = torch.nn.CrossEntropyLoss()
            loss = celoss(output, target)

            # interpret softmax as set of label predictions
            _, predictions = torch.max(output[:], 1)

            # Compute gradients
            learner.zero_grad()
            loss.backward()

            grad = torch.cat([p.grad.data.view(-1) / BATCH_SIZE for p in learner.parameters()], 0)

            # preprocess the gradients of the learner to handle varying magnitudes
            pgrad = preprocess_parameters(grad)
            ploss = preprocess_parameters(loss.data.unsqueeze(0))

            metalearner_input = [ploss, pgrad, grad.unsqueeze(1)]
            memory_cell, new_h = metalearner(metalearner_input, hidden_states[-1])
            hidden_states.append(new_h)

    return memory_cell


def meta_test(val_dataset, learner, learner_wo_grad, metalearner):
    best_acc = 0
    for _ in range(VAL_ITERATIONS):
        # Get the data to train and test the model
        x, y = val_dataset.get_item()

        train_x = x[:, :SHOTS].reshape((SHOTS * CLASSES,) + x.shape[2:]).to(device)
        train_y = y[:, :SHOTS].flatten()

        test_x = x[:, SHOTS:].reshape((EVALS * CLASSES,) + x.shape[2:]).to(device)
        test_y = torch.LongTensor(y[:, SHOTS:].flatten()).to(device)

        # Reset the batch norm layers
        learner.reset_batch_norm()
        learner_wo_grad.reset_batch_norm()

        # Set the Laerner with gradients to train mode and the learner without gradients to eval
        learner.train()
        learner_wo_grad.eval()

        # Get and clone the updated parameters for the learner as well as the running stats of the other learner,
        # connecting the backprop computation graph
        cell_state = train_learner(learner, metalearner, train_x, train_y)
        stats = learner.get_bn_stats()
        learner_wo_grad.clone_flat_params(cell_state, stats)

        # Get validation set predictions and return accuracy
        output = learner_wo_grad(test_x)
        preds = torch.max(output[:], 1)[1]

        current_acc = accuracy(preds, truth=test_y)

        if current_acc > best_acc:
            best_acc = current_acc

    return best_acc


def main():
    # Transforms for preprocessing the data
    if INPUT_DIM == 3:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_set_transform = transforms.Compose([transforms.RandomResizedCrop(CROPPED_IMAGE_SIZE),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                                                                         hue=0.2),
                                                  transforms.ToTensor(), normalize])
        val_set_transform = transforms.Compose([transforms.Resize(CROPPED_IMAGE_SIZE * 8 // 7),
                                                transforms.CenterCrop(CROPPED_IMAGE_SIZE),
                                                transforms.ToTensor(), normalize])

    elif INPUT_DIM == 1:
        normalize = transforms.Normalize([0.5], [0.5])
        train_set_transform = transforms.Compose([transforms.RandomResizedCrop(CROPPED_IMAGE_SIZE),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                                                                         hue=0.2),
                                                  transforms.Grayscale(),
                                                  transforms.ToTensor(), normalize])
        val_set_transform = transforms.Compose([transforms.Resize(CROPPED_IMAGE_SIZE * 8 // 7),
                                                transforms.CenterCrop(CROPPED_IMAGE_SIZE),
                                                transforms.Grayscale(),
                                                transforms.ToTensor(), normalize])

    if args.data == 'MIN':  # miniImagenet
        train_dataset = data_loader.MetaMINDataset(TRAIN_PATH, SHOTS, EVALS, CLASSES, train_set_transform,
                                                   CROPPED_IMAGE_SIZE)
        val_dataset = data_loader.MetaMINDataset(VAL_PATH, SHOTS, EVALS, CLASSES, val_set_transform, CROPPED_IMAGE_SIZE)
        learner = CNNlearner.CNNLearner(CROPPED_IMAGE_SIZE, FILTERS, KERNEL_SIZE, OUTPUT_DIM, BN_MOMENTUM,
                                        in_channels=INPUT_DIM).to(device)

    if args.data == 'MNIST':
        train_dataset = data_loader.MetaMNISTDataset(TRAIN_PATH, SHOTS, EVALS, CLASSES)
        val_dataset = data_loader.MetaMNISTDataset(VAL_PATH, SHOTS, EVALS, CLASSES)
        learner = CNNlearner.CNNLearner(CROPPED_IMAGE_SIZE, FILTERS, KERNEL_SIZE, OUTPUT_DIM, BN_MOMENTUM,
                                        in_channels=INPUT_DIM).to(
            device)

    # Learner without gradient history
    grad_free_learner = copy.deepcopy(learner)
    grad_free_learner = grad_free_learner.to(device)

    metalearner = meta_learner.MetaLearner(INPUT_SIZE, HIDDEN_SIZE, learner.get_flat_params().size(0)).to(device)
    metalearner.init_memory_cell(learner.get_flat_params())

    optimiser = torch.optim.Adam(metalearner.parameters(), lr=LEARNING_RATE)
    learner_loss_function = torch.nn.CrossEntropyLoss().to(device)

    if args.test:
        if args.load_state != None:
            metalearner.load_state_dict(torch.load(args.load_state))

        best_acc = meta_test(val_dataset, learner, grad_free_learner, metalearner)
        print("Best Accuracy {:.4f} | Time: {}".format(best_acc, time.time()))
        quit()

    # Training Loop
    for it in range(ITERATIONS):
        x, y = train_dataset.get_item()

        train_x = x[:, :SHOTS].reshape((SHOTS * CLASSES,) + x.shape[2:]).to(device)
        train_y = y[:, :SHOTS].flatten()

        test_x = x[:, SHOTS:].reshape((EVALS * CLASSES,) + x.shape[2:]).to(device)
        test_y = torch.LongTensor(y[:, SHOTS:].flatten()).to(device)

        # Train learner with metalearner
        learner.reset_batch_norm()
        grad_free_learner.reset_batch_norm()
        learner.train()
        grad_free_learner.train()
        new_cell_state = train_learner(learner, metalearner, train_x, train_y)

        # Clone the new parameters, so the computation graph is connected.
        stats = learner.get_bn_stats()
        grad_free_learner.clone_flat_params(new_cell_state, stats)

        output = grad_free_learner(test_x)
        predictions = torch.max(output[:], 1)[1]
        loss = learner_loss_function(output, test_y)
        train_acc = accuracy(predictions, test_y)
        train_acc_hist.append(train_acc)
        print("Iteration {} | Training Accuracy {:.4f} | Time: {}".format(it, train_acc, time.time()))

        optimiser.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(metalearner.parameters(), GRADIENT_CLIPPING)
        optimiser.step()

        # Meta-validation
        if it % EVAL_POINT == 0 or it == (ITERATIONS - 1):
            torch.save(metalearner.state_dict(), args.save_state)
            val_acc = meta_test(val_dataset, learner, grad_free_learner, metalearner)
            print("Iteration {} | Training Accuracy {:.4f} | Best Validation Accuracy {:.4f} | Time: {}".
                  format(it, train_acc, val_acc, time.time()))
            val_acc_hist.append(val_acc)

            print("Avg. Train Accuracy: " + str(np.mean(np.array(train_acc_hist))))
            print("Avg. Validation Accuracy: " + str(np.mean(np.array(val_acc_hist))))


if __name__ == "__main__":
    main()

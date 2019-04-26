from data_loader import MetaDataset

iterations = 50000
evals = 15 # items used to test acc and loss
classes = 5 # number of classes we differentiate
shots = 5 # items used to train with for each class
train_path = "train"
test_path = "test"
val_path = "val"

def train_meta_learner(train_dataset, iterations):
    for _ in range(iterations):
        x, y = train_dataset.getitem()

        train_reshape_form = (shots, shots + evals) + x.shape[1:]
        x = x.reshape(train_reshape_form)
        train_x = x[:, :shots].reshape((shots * classes, ) + x.shape[2:])
        test_x =  x[:, shots:].reshape((evals * classes, ) + x.shape[2:])



train_dataset = MetaDataset(train_path, shots, evals, classes)
test_dataset = MetaDataset(test_path, shots, evals, classes)

train_meta_learner(train_dataset, iterations)
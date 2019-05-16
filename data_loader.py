import os
import torch
import random
import torchvision
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms


class MetaMINDataset:
    def __init__(self, root_dir, shots, evals, no_classes, transform=None, crop=128):
        """
        Creates a new instance of the MiniImageNet MetaDataset class
        :param root_dir: string, images root directory
        :param shots: int k, for the k-shot classification task (i.e., 1-shot, 5-shot)
        :param evals: int, number of examples to use for testing purposes
        :param no_classes: int, number of classes to work with
        :param crop: int, square dim to crop images to
        """
        self.root_dir = root_dir
        self.shots = shots
        self.evals = evals
        self.no_classes = no_classes

        self.transform = transform
        if transform is None:
            # transforms.Compose() object to be applied to each image
            self.transform = transforms.Compose([transforms.RandomResizedCrop(crop), transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.crop = crop

        # Pre-load all images to memory for speed
        self.all_images = self.__load_images__()

    def get_item(self):
        """
        :return: return no_classes * (shots + evals) tensors with corresponding labels
        """
        sampled_class_names = random.sample(self.all_images.keys(), self.no_classes)
        sample_item_count = self.shots + self.evals

        sampled_images = []

        for name in sampled_class_names:
            current_sample = random.sample(self.all_images[name], sample_item_count)
            sampled_images.extend(current_sample)

        # labels only have to indicate how tensors group into different classes
        labels = np.repeat(range(self.no_classes), self.shots + self.evals)

        data = torch.stack(sampled_images)

        data = data.reshape((self.no_classes, self.shots + self.evals) + data.shape[1:])
        labels = labels.reshape((self.no_classes, self.shots + self.evals))

        return data, labels

    def __load_images__(self):
        """
        Loads the entire dataset to memory
        :return: a dictionary containing the class labels as
                keys and a list of Tensors (one per image)
        """
        class_dirs = [name for name in os.listdir(self.root_dir)]
        all_images = {}
        for dir in class_dirs:
            class_images = []

            current_path = os.path.join(self.root_dir, dir)
            image_names = os.listdir(current_path)

            for name in image_names:
                image_path = os.path.join(current_path, name)

                image = Image.open(image_path)
                image_tensor = self.transform(image)
                class_images.append(image_tensor)

            all_images[dir] = class_images

        return all_images


class MetaMNISTDataset:
    def __init__(self, root_dir, shots, evals, no_classes, transform=None, crop=84):
        self.root_dir = root_dir
        self.shots = shots
        self.evals = evals
        self.no_classes = no_classes
        self.transform = transform

        if transform is None:
        # transforms.Compose() object to be applied to each image
            self.transform = torchvision.transforms.Compose([
                transforms.Resize((crop, crop)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

        # obtain the mnist dataset
        self.loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root_dir, download=True, transform=self.transform),
            batch_size=(shots+evals)*no_classes * 3, shuffle=False)
        self.examples = enumerate(self.loader)

    def get_item(self):
        # require shots + evals random images from each class
        while True:
            batch_idx, (example_data, example_targets) = next(self.examples)
            x_unique = example_targets.unique(sorted=True)
            x_unique_count = torch.stack([(example_targets == x_u).sum() for x_u in x_unique])
            if all(i >= (self.shots + self.evals) for i in x_unique_count):
                break

        sort_by_label = [x for _, x in sorted(zip(example_targets.tolist(), example_data.tolist()))]
        x = np.reshape(sort_by_label, ((self.shots+self.evals)*self.no_classes * 3, 1, 84, 84))
        trainlist = []
        for i in range(10):
            if i > 0:
                i = sum(x_unique_count[:i])
            trainlist.append(x[i:i + self.shots + self.evals])

        trainlist = np.array(trainlist)

        labels = np.repeat(range(self.no_classes), self.shots + self.evals)
        return torch.from_numpy(trainlist), labels.reshape(self.no_classes, self.shots + self.evals)

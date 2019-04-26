import os
import random

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms


class MetaDataset:
    def __init__(self, root_dir, shots, evals, no_classes, crop=128,
                 transform=transforms.Compose([transforms.RandomResizedCrop(128),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        """
        :param root_dir: directory containing folders for each class
        :param shots: number items used for training
        :param evals: number of items used for evaluation to get acc and loss
        :param no_classes: number of classes to learn to differentiate
        :param crop: size to crop image down to
        :param transform: transform function for processing images
        """
        self.root_dir = root_dir
        self.shots = shots
        self.evals = evals
        self.no_classes = no_classes
        self.transform = transform
        self.crop = crop

    def getitem(self):
        """

        :return: return no_classes * (shots + evals) tensors with corresponding labels
        """
        # pick classes number of random directories
        class_dirs = [name for name in os.listdir(self.root_dir)]
        total_classes = len(class_dirs)
        sample_classes = random.sample(range(1, total_classes), self.no_classes)

        sample_item_count = self.shots + self.evals

        sample_items = []

        # read shots + eval random items for each of those classes
        for i in range(self.no_classes):
            # create path for this class
            path = os.path.join(self.root_dir, class_dirs[sample_classes[i]])
            item_names = os.listdir(path)
            no_items = len(item_names)
            sample_indexes = random.sample(range(1, no_items), sample_item_count)

            for j in range(sample_item_count):
                sample_item_name = item_names[sample_indexes[j]]

                full_path = os.path.join(path, sample_item_name)
                image = Image.open(full_path)
                image_tensor = self.transform(image)
                sample_items.append(image_tensor)

        # labels only have to indicate how tensors group into different classes
        labels = np.repeat(range(self.no_classes), self.shots)
        return torch.stack(sample_items), labels

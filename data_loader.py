import os
import torch
import random
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms


class MetaDataset:
    def __init__(self, root_dir, shots, evals, no_classes, crop=128,
                 transform=transforms.Compose([transforms.RandomResizedCrop(128),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        """
        Creates a new instance of the MetaDataset class
        :param root_dir: string, images root directory
        :param shots: int k, for the k-shot classification task (i.e., 1-shot, 5-shot)
        :param evals: int, number of examples to use for testing purposes
        :param no_classes: int, number of classes to work with
        :param crop: int, square dim to crop images to
        :param transform: transforms.Compose() object to be applied to each image
        """
        self.root_dir = root_dir
        self.shots = shots
        self.evals = evals
        self.no_classes = no_classes
        self.transform = transform
        self.crop = crop

    def getitem(self):

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

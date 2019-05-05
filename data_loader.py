import os
import torch
import random
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms


class MetaDataset:
    def __init__(self, root_dir, shots, evals, no_classes, crop=128):
        """
        Creates a new instance of the MetaDataset class
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

        # transforms.Compose() object to be applied to each image
        self.transform = transforms.Compose([transforms.RandomResizedCrop(128), transforms.ToTensor(),
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

        return torch.stack(sampled_images), labels

    def __load_images__(self):
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

import os
import random

import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image


def get_image_list(root):
    # images = []
    # for class_dir in os.listdir(root):
    #     for image in os.listdir(os.path.join(root, class_dir)):
    #         image_path = os.path.join(root, class_dir, image)
    #         images.append(image_path)
    # return images
    with open(root, 'r') as f:
        file_list = f.readlines()
        file_list = [row.rstrip().split(" ")[0] for row in file_list]
    
    return file_list


def color_projection(image):
    new_image = torch.zeros_like(image)
    r, g, b = image[0], image[1], image[2]
    new_image[0] = 0.8333 * r + 0.3333 * g - 0.1667 * b
    new_image[1] = 0.3333 * r + 0.3333 * g + 0.3333 * b
    new_image[2] =-0.1667 * r + 0.3333 * g + 0.8333 * b
    return new_image


class JigsawDataset(data.Dataset):
    def __init__(self, root, perms_file):
        self.root = root
        self.perms_file = perms_file
        self.perms = np.load(perms_file)
        self.classes = list(range(self.perms.shape[0]))
        self.images = get_image_list(root)
        self.resize = transforms.Resize(256)
        self.rand_crop = transforms.RandomCrop(255)
        self.grayscale = transforms.Grayscale(num_output_channels=3)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        image_path = self.images[index]
        target = np.random.randint(0, self.perms.shape[0])
        image = default_loader(image_path)
        perm = self.perms[target]

        image = self.rand_crop(self.resize(image))
        if random.random() < 0.5:
            image = self.grayscale(image)
            image = self.normalize(self.to_tensor(image))
        else:
            image = self.to_tensor(image)
            if random.random() < 0.5:
                image = color_projection(image)
            image = self.normalize(image)

        k = 0
        tiles = torch.zeros((9, 3, 85, 85), dtype=image.dtype)
        for i in range(0, 255, 85):
            for j in range(0, 255, 85):
                tiles[k] = image[:, i:i+85, j:j+85]
                k += 1

        k = 0
        shuffled_tiles = torch.zeros((9, 3, 64, 64), dtype=image.dtype)
        for i in perm:
            x = np.random.randint(0, 21)
            y = np.random.randint(0, 21)
            shuffled_tiles[i] = tiles[k, :, x:x+64, y:y+64]
            k += 1

        return shuffled_tiles, target

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        if self.transform is not None:
            images = self.transform(img)

        return images, target

    def __len__(self):
        return len(self.file_list)

def denormalize(tensor):
    tensor = tensor.cpu()
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    array = ((tensor * std[:, None, None]) + mean[:, None, None]).numpy()
    return array.transpose((1, 2, 0))


def show_jigsaw(shuffled_tiles, image_path, target):
    image_name = image_path.split('/')[-1]
    bn, ext = image_name.split('.')
    image_name = '%s_%d.%s' % (bn, target, ext)
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for i in range(3):
        for j in range(3):
            axes[i][j].imshow(denormalize(shuffled_tiles[(i * 3) + j]))
    fig.savefig(os.path.join('example_puzzles', image_name))

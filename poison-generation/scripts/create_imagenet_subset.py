'''
Author: Aniruddha Saha
Description: This scripts creates the Imagenet100 subset.
                Instead of using symlinks like in the original repo, this copies the images 
                for ease of use in the Dataset class in this repo which loads images from lists.
'''
import os
import re
import errno
import argparse
import shutil
from tqdm import tqdm


def create_subset(class_list, full_imagenet_path, subset_imagenet_path, *,
                  splits=('train', 'val')):
    full_imagenet_path = os.path.abspath(full_imagenet_path)
    subset_imagenet_path = os.path.abspath(subset_imagenet_path)
    os.makedirs(subset_imagenet_path, exist_ok=True)
    for split in splits:
        os.makedirs(os.path.join(subset_imagenet_path, split), exist_ok=True)
    for c in tqdm(class_list):
        if re.match(r"n[0-9]{8}", c) is None:
            raise ValueError(
                f"Expected class names to be of the format nXXXXXXXX, where "
                f"each X represents a numerical number, e.g., n04589890, but "
                f"got {c}")
        for split in splits:
            shutil.copytree(
                os.path.join(full_imagenet_path, split, c),
                os.path.join(subset_imagenet_path, split, c)
            )
    print(f'Finished creating ImageNet subset at {subset_imagenet_path}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Subset Creation')
    parser.add_argument('--full_imagenet_path', metavar='IMAGENET_DIR',
                        help='path to the existing full ImageNet dataset')
    parser.add_argument('--subset_imagenet_path', metavar='SUBSET_DIR',
                        help='path to create the ImageNet subset dataset')
    parser.add_argument('--subset', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'imagenet100_classes.txt'),
                        help='file contains a list of subset classes')
    args = parser.parse_args()

    print(f'Using class names specified in {args.subset}.')
    with open(args.subset, 'r') as f:
        class_list = [l.strip() for l in f.readlines()]

    create_subset(class_list, args.full_imagenet_path, args.subset_imagenet_path)

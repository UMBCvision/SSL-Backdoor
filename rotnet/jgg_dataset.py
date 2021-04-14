import torch.utils.data as data

from PIL import Image

import os
import os.path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_entries(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    entries = [line.strip().split(' ') for line in lines]
    entries = [(pth, int(label)) for pth, label in entries]
    return entries


class JGGDataset(data.Dataset):
    def __init__(self, root, txt_file, transform=None, target_transform=None, loader=default_loader):
        self.entries = [(os.path.join(root, pth), label - 1)
                        for pth, label in get_entries(txt_file)]
        self.classes = list(range(2000))
        self.root = root
        self.loader = loader

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.entries[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


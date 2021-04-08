import random
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import ImageFilter, Image
from .transforms import MultiSample, aug_transform
from .base import BaseDataset
from torch.utils import data

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

class RandomBlur:
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1

    def __call__(self, image):
        r = random.uniform(self.r0, self.r1)
        return image.filter(ImageFilter.GaussianBlur(radius=r))


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

# Change this to be consistent with MoCo v2 eval
def base_transform_eval():
    return T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )


class ImageNet(BaseDataset):
    def ds_train(self):
        aug_with_blur = aug_transform(
            224,
            base_transform,
            self.aug_cfg,
            extra_t=[T.RandomApply([RandomBlur(0.1, 2.0)], p=0.5)],
        )
        t = MultiSample(aug_with_blur, n=self.aug_cfg.num_samples)
        return FileListDataset(path_to_txt_file=self.aug_cfg.train_file_path, transform=t)

    # Do not pre resize images like in original repo
    def ds_clf(self):
        t = base_transform_eval()
        return FileListDataset(path_to_txt_file=self.aug_cfg.train_clean_file_path, transform=t)

    def ds_test(self):
        t = base_transform_eval()
        return FileListDataset(path_to_txt_file=self.aug_cfg.val_file_path, transform=t)

    def ds_test_p(self):
        # val poisoned images are already preprocessed
        t = base_transform()
        return FileListDataset(path_to_txt_file=self.aug_cfg.val_poisoned_file_path, transform=t)

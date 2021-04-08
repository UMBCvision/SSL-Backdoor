import os
from torch.utils import data
from PIL import Image

class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform):
        # self.data_root = data_root
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

        return image_path, images, target, idx

    def __len__(self):
        return len(self.file_list)

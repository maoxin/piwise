import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision.transforms import Resize, ColorJitter, CenterCrop, RandomCrop, Normalize, RandomHorizontalFlip
from torchvision.transforms import ToTensor
from piwise.transform import ToLabel

import random

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class ADE(Dataset):

    def __init__(self, root):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        
        image = Resize((256, 256))(image)
        image = ColorJitter(brightness=0.5)(image)
        
        label = Resize((256, 256), Image.NEAREST)(label)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        image = RandomHorizontalFlip()(image)
        random.seed(seed)
        label = RandomHorizontalFlip()(label)
        

        # if_lr = np.random.choice([False, True])

        # if if_lr:
            # image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # label = label.transpose(Image.FLIP_LEFT_RIGHT)

        image = ToTensor()(image)
        image = Normalize([.485, .456, .406], [.229, .224, .225])(image)
        label = ToLabel()(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class ADE_Val(Dataset):

    def __init__(self, root):
        self.images_root = os.path.join(root, 'img_val')
        self.labels_root = os.path.join(root, 'lbl_val')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        
        image = Resize((256, 256))(image)
        label = Resize((256, 256), Image.NEAREST)(label)

        image = ToTensor()(image)
        image = Normalize([.485, .456, .406], [.229, .224, .225])(image)
        label = ToLabel()(label)

        return image, label

    def __len__(self):
        return len(self.filenames)
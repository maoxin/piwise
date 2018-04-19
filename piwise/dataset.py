import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision.transforms import Resize, ColorJitter, CenterCrop
from torchvision.transforms import ToTensor
from piwise.transform import ToLabel

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

        # short_size = int(np.random.choice([300, 375, 450, 525, 600]))
        short_size = int(np.random.choice([256]))

        # image = self.resize(image, short_size)
        image = CenterCrop(256)(image)
        image = ColorJitter(brightness=0.5)(image)

        # label = self.resize(label, short_size)
        label = CenterCrop(256)(label)

        if_lr = np.random.choice([False, True])
        if_td = np.random.choice([False, True])

        # if if_lr:
            # image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # label = label.transpose(Image.FLIP_LEFT_RIGHT)
        # if if_td:
            # image = image.transpose(Image.FLIP_TOP_BOTTOM)
            # label = label.transpose(Image.FLIP_TOP_BOTTOM)

        image = ToTensor()(image)
        label = ToLabel()(label)

        # print('image', image.size())
        # print('label', label.size())

        return image, label

    def __len__(self):
        return len(self.filenames)

    def resize(self, img, short_size):
        img = Resize(short_size)(img)
        
        target_height = self.round2nearest_multiple(img.height)
        target_width = self.round2nearest_multiple(img.width)
        img = Resize((target_height, target_width))(img)

        return img

    def round2nearest_multiple(self, x, p=16):
        return ((x - 1) // p + 1) * p
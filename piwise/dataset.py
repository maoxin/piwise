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

        # short_size = int(np.random.choice([304, 384, 464, 528]))

        # image = self.resize_img(image, short_size)
        # image = self.resize_img(image) # only when batch_size=1
        image = CenterCrop(256)(image)
        image = ColorJitter(brightness=0.5)(image)

        # label = self.resize_label(label, short_size)
        # label = self.resize_label(label)
        label = CenterCrop(256)(label)

        if_lr = np.random.choice([False, True])
        if_td = np.random.choice([False, True])

        # if if_lr:
        #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #     label = label.transpose(Image.FLIP_LEFT_RIGHT)
        # if if_td:
        #     image = image.transpose(Image.FLIP_TOP_BOTTOM)
        #     label = label.transpose(Image.FLIP_TOP_BOTTOM)

        image = ToTensor()(image)
        label = ToLabel()(label)

        # print('image', image.size())
        # print('label', label.size())

        return image, label

    def __len__(self):
        return len(self.filenames)

    def resize_img(self, img, short_size=None):
        if short_size:
            if img.height < img.width:
                target_height0 = short_size
                target_width0 = int(short_size / target_height0 * img.width)
            else:
                target_width0 = short_size
                target_height0 = int(short_size / target_width0 * img.height)
            img = Resize((target_height0, target_width0))(img)
        
        target_height = self.round2nearest_multiple(img.height)
        target_width = self.round2nearest_multiple(img.width)

        if (max([target_height, target_width]) <= 784):
            img = Resize((target_height, target_width))(img)
        else:
            img = CenterCrop(528)(img)

        return img

    def resize_label(self, img, short_size=None):
        if short_size:
            if img.height < img.width:
                target_height0 = short_size
                target_width0 = int(short_size / target_height0 * img.width)
            else:
                target_width0 = short_size
                target_height0 = int(short_size / target_width0 * img.height)
            img = Resize((target_height0, target_width0), interpolation=Image.NEAREST)(img)
        
        target_height = self.round2nearest_multiple(img.height)
        target_width = self.round2nearest_multiple(img.width)

        if (max([target_height, target_width]) <= 784):
            img = Resize((target_height, target_width), interpolation=Image.NEAREST)(img)
        else:
            img = CenterCrop(528)(img)

        return img

    def round2nearest_multiple(self, x, p=16):
        return ((x - 1) // p + 1) * p
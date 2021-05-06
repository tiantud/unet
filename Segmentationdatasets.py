import os
import time
import random
import torch
import numpy as np

from PIL import Image

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string):                  path to a file
        extensions (tuple of strings):      extensions to consider (lowercase)
    Returns:
        bool:                               True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string):                  path to a file
    Returns:
        bool:                               True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


class SegmentationDatasetFolder():
    """A generic data loader

    Args:
        img_tuple_list (tuple of path):     list of tuple, each has form (path_to_ori_img, path_to_mask_img)
        transform (callable, optional):     A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional):        A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list):                     List of the class names.
        class_to_idx (dict):                Dict with items (class_name, class_index).
        imgs (list):                        List of (image path, class_index) tuples
    """

    def __init__(self, img_tuple_list, loader, extensions=None, input_transform=None, output_transform=None, is_valid_file=None, 
        sample_channel=3, mask_channel=3):

        assert len(img_tuple_list) != 0, "Found 0 files."

        self.input_transform = input_transform
        self.output_transform = output_transform

        self.loader = loader
        self.extensions = extensions

        self.sample_channel = sample_channel
        self.mask_channel = mask_channel

        self.samples = img_tuple_list

    def __getitem__(self, index):
        """
        Args:
            index (int):                    Index
        Returns:
            tuple (sample, mask):           where mask is class_index of the mask_path class.
        """
        (path, mask_path) = self.samples[index]
        sample = self.loader(path, n_channel=self.sample_channel)
        mask = self.loader(mask_path, n_channel=self.mask_channel)
        fname = os.path.split(path)[1]

        if self.input_transform or self.output_transform:
            timeseed = int(time.time())
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.input_transform:
            torch.manual_seed(timeseed)
            torch.cuda.manual_seed_all(timeseed)
            np.random.seed(timeseed)
            random.seed(timeseed)
            sample = self.input_transform(sample)

        if self.output_transform:
            torch.manual_seed(timeseed)
            torch.cuda.manual_seed_all(timeseed)
            np.random.seed(timeseed)
            random.seed(timeseed)
            mask = self.output_transform(mask)

        return sample, mask, fname

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path, *, n_channel):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') if n_channel == 3 else img.convert('L')


def accimage_loader(path, *, n_channel):
    import accimage
    try:
        img = accimage.Image(path)
        return img.convert('RGB') if n_channel == 3 else img.convert('L')
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path, n_channel=n_channel)


def default_loader(path, *, n_channel):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path, n_channel=n_channel)


class SegmentationImageFolder(SegmentationDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/group_1/xxx.jpg
        root/group_1/xxx_mask.jpg
        root/group_1/xxz.jpg
        root/group_1/xxz_mask.jpg

        root/group_n/123.jpg
        root/group_n/123_mask.jpg

    Args:
        root (string):                      list of tuple, each has form (path_to_ori_img, path_to_mask_img)
        transform (callable, optional):     A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional):        A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list):                     List of the class names.
        class_to_idx (dict):                Dict with items (class_name, class_index).
        imgs (list):                        List of (image path, class_index) tuples
    """

    def __init__(self, img_tuple_list, input_transform=None, output_transform=None, loader=default_loader, is_valid_file=None,
        sample_channel=3, mask_channel=3):

        super(SegmentationImageFolder, self).__init__(img_tuple_list, loader, 
            IMG_EXTENSIONS if is_valid_file is None else None,
            input_transform=input_transform,
            output_transform=output_transform,
            is_valid_file=is_valid_file,
            sample_channel=sample_channel, 
            mask_channel=mask_channel)

        self.imgs = self.samples
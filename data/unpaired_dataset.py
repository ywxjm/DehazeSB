"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import os
import re
import numpy as np
import random
import scipy
import torch
import torch.utils.data
from PIL import Image
from PIL import ImageOps
import util.util as util
import pandas as pd
import torchvision.transforms as T

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def lstFiles(path):
    image_list=[]
    for image in os.listdir(path):
        if ".png" in image.lower():
            image_list.append(os.path.join(path, image))
        elif ".jpeg" in image.lower():
            image_list.append(os.path.join(path, image))
        elif ".jpg" in image.lower():
            image_list.append(os.path.join(path, image))

    image_list = sorted(image_list, key=numericalSort)
    return image_list

def Normalization(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val != 0:
        image = (image - min_val) / (max_val - min_val) * 2.0 -1.0
    else:
        image = np.zeros_like(image)


    return image



class ValidationSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(ValidationSet,self).__init__()
        opt.new_image_size = 512
        # self.dir_A = os.path.join(opt.validation_dict_path, 'test_real_hazy')  # create a path to validation set
        self.dir_B = os.path.join('/data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_new_v2_delete_bluesky/test_real_hazy_det/Images')
        self.dir_A = os.path.join('/data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_new_v2_delete_bluesky/test_real_hazy_det/Images')

        self.A_paths=lstFiles(self.dir_A) ## the list that store the absolute path for each images
        self.B_paths=lstFiles(self.dir_B)

        #####
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.size = opt.new_image_size ## create it load training dataloader
        ##
        self.transform = T.Compose([
            # T.Resize((self.size, self.size)),  # Resize to 256x256
            T.ToTensor(),  # Convert back to tensor (C, H, W) and normalize to (0,1)
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ## normalize to -1 and 1
        ])
        # self.high_quality_path = opt.validation_dict_path

    def read_image(self, path):
        # Open the image and convert to RGB (for JPEG support)
        readed_image = Image.open(path).convert("RGB")
        if readed_image.height >= 1000 or readed_image.width >= 1000:
            readed_image = readed_image.resize((1024, 1024), Image.LANCZOS)

        else:
            new_h = (readed_image.height // 8) * 8
            new_w = (readed_image.width // 8) * 8
            readed_image = readed_image.resize((new_w, new_h), Image.LANCZOS)

        return readed_image

    ####
    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):  ## need to return the A(LQ) with regard to related ground truth B

        A_path = self.A_paths[index % self.A_size]  ## find absolte A_path
        image_name = A_path.split('/')[-1] ##10001_right_001.jpeg  ## baseline : 10001.right.png
        A_img = self.read_image(A_path)
        A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor and normalize to -1 to 1

        B_img = A_img
        B_path = A_path

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}




class ValidationHaze2020(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(ValidationHaze2020, self).__init__()
        opt.new_image_size = 512
        self.dir_A = os.path.join('/data/lanyunwei/my_experiment_results/dehaze_demo_results/hazy/hazy2020/')  # create a path to validation set
        self.dir_B = os.path.join('/data/lanyunwei/my_experiment_results/dehaze_demo_results/hazy/hazy2020/')

        self.A_paths = lstFiles(self.dir_A)  ## the list that store the absolute path for each images
        self.B_paths = lstFiles(self.dir_B)

        #####
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.size = opt.new_image_size  ## create it load training dataloader
        ##
        self.transform = T.Compose([
            # T.Resize((self.size, self.size)),  # Resize to 256x256
            T.ToTensor(),  # Convert back to tensor (C, H, W) and normalize to (0,1)
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  ## normalize to -1 and 1
        ])
        # self.high_quality_path = opt.validation_dict_path

    def read_image(self, path):
        # Open the image and convert to RGB (for JPEG support)
        readed_image = Image.open(path).convert("RGB")
        if readed_image.height >= 1000 or readed_image.width >= 1000:
            readed_image = readed_image.resize((1024, 1024), Image.LANCZOS)

        else:
            new_h = (readed_image.height // 8) * 8
            new_w = (readed_image.width // 8) * 8
            readed_image = readed_image.resize((new_w, new_h), Image.LANCZOS)

        return readed_image

        ####

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):  ## need to return the A(LQ) with regard to related ground truth B

        A_path = self.A_paths[index % self.A_size]  ## find absolte A_path
        image_name = A_path.split('/')[-1]  ##10001_right_001.jpeg  ## baseline : 10001.right.png
        A_img = self.read_image(A_path)
        A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor and normalize to -1 to 1

        B_img = A_img
        B_path = A_path

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

class ValidationOHAZE(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(ValidationOHAZE, self).__init__()
        opt.new_image_size = 512
        self.dir_A = os.path.join('/data/lanyunwei/my_experiment_results/O-HAZE/test_B/')  # create a path to validation set
        self.dir_B = os.path.join('/data/lanyunwei/my_experiment_results/O-HAZE/test_B/')

        self.A_paths = lstFiles(self.dir_A)  ## the list that store the absolute path for each images
        self.B_paths = lstFiles(self.dir_B)

        #####
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.size = opt.new_image_size  ## create it load training dataloader
        ##
        self.transform = T.Compose([
            # T.Resize((self.size, self.size)),  # Resize to 256x256
            T.ToTensor(),  # Convert back to tensor (C, H, W) and normalize to (0,1)
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  ## normalize to -1 and 1
        ])
        # self.high_quality_path = opt.validation_dict_path

    def read_image(self, path):
        # Open the image and convert to RGB (for JPEG support)
        readed_image = Image.open(path).convert("RGB")
        if readed_image.height >= 1000 or readed_image.width >= 1000:
            readed_image = readed_image.resize((1024, 1024), Image.LANCZOS)

        else:
            new_h = (readed_image.height // 8) * 8
            new_w = (readed_image.width // 8) * 8
            readed_image = readed_image.resize((new_w, new_h), Image.LANCZOS)



        return readed_image

        ####

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):  ## need to return the A(LQ) with regard to related ground truth B

        A_path = self.A_paths[index % self.A_size]  ## find absolte A_path
        image_name = A_path.split('/')[-1]  ##10001_right_001.jpeg  ## baseline : 10001.right.png
        A_img = self.read_image(A_path)
        A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor and normalize to -1 to 1

        B_img = A_img
        B_path = A_path

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}


class ValidationURHI(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(ValidationURHI, self).__init__()
        opt.new_image_size = 512
        self.dir_A = os.path.join('/data/lanyunwei/URHI/UnannotatedHazyImages/')  # create a path to validation set
        self.dir_B = os.path.join('/data/lanyunwei/my_experiment_results/O-HAZE/test_B/')

        self.A_paths = lstFiles(self.dir_A)  ## the list that store the absolute path for each images
        self.B_paths = lstFiles(self.dir_B)

        #####
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.size = opt.new_image_size  ## create it load training dataloader
        ##
        self.transform = T.Compose([
            # T.Resize((self.size, self.size)),  # Resize to 256x256
            T.ToTensor(),  # Convert back to tensor (C, H, W) and normalize to (0,1)
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  ## normalize to -1 and 1
        ])
        # self.high_quality_path = opt.validation_dict_path

    def read_image(self, path):
        # Open the image and convert to RGB (for JPEG support)
        readed_image = Image.open(path).convert("RGB")
        if readed_image.height >= 1000 or readed_image.width >= 1000:
            readed_image = readed_image.resize((1024, 1024), Image.LANCZOS)

        else:
            new_h = (readed_image.height // 8) * 8
            new_w = (readed_image.width // 8) * 8
            readed_image = readed_image.resize((new_w, new_h), Image.LANCZOS)



        return readed_image

        ####

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):  ## need to return the A(LQ) with regard to related ground truth B

        A_path = self.A_paths[index % self.A_size]  ## find absolte A_path
        image_name = A_path.split('/')[-1]  ##10001_right_001.jpeg  ## baseline : 10001.right.png
        A_img = self.read_image(A_path)
        A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor and normalize to -1 to 1

        B_img = A_img
        B_path = A_path

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}


class UnpairedDataSet(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_image_size', type=int, default=256, help='new dataset size')
        return parser

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)


        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_hazy_det', 'Images')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_clear_det', 'Images')

        self.A_paths=lstFiles(self.dir_A) ## the list that store the absolute path for each images
        self.B_paths=lstFiles(self.dir_B)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.size = opt.new_image_size

        self.transform = T.Compose([
            T.Resize((self.size, self.size)),  # Resize to 256x256
            T.ToTensor(),  # Convert back to tensor (C, H, W)
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ## normalize to -1 and 1
        ])

    def read_image(self, path):
        readed_image = Image.open(path).convert("RGB")
        return readed_image

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = self.read_image(A_path)
        B_img = self.read_image(B_path)

        A_img = self.transform(A_img)  # Convert (H, W, C) -> (C, H, W) Tensor
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)




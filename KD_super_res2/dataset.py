import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch

class SR_SliceData(Dataset):

    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self,root):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        self.mask = np.zeros([2,256,256])
        self.mask[:,128-32:128+32,128-32:128+32] = 1

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:

            target = data['volfs'][slice,:,:,:]
            target = np.transpose(target,[2,0,1])
            kspace = np.fft.fft2(target,norm='ortho')
            kspace_shifted = np.fft.fftshift(kspace)
            truncated_kspace = self.mask * kspace_shifted

            lr_img = np.abs(np.fft.ifft2(truncated_kspace,norm='ortho'))

            return torch.from_numpy(lr_img),torch.from_numpy(target)

class SR_SliceDataDev(Dataset):

    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self,root):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        self.mask = np.zeros([2,256,256])
        self.mask[:,128-32:128+32,128-32:128+32] = 1

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:

            target = data['volfs'][slice,:,:,:]
            target = np.transpose(target,[2,0,1])
            kspace = np.fft.fft2(target,norm='ortho')
            kspace_shifted = np.fft.fftshift(kspace)
            truncated_kspace = self.mask * kspace_shifted

            lr_img = np.abs(np.fft.ifft2(truncated_kspace,norm='ortho'))

            return torch.from_numpy(lr_img),torch.from_numpy(target),str(fname.name),slice


################################################################################################################################# 

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, acc_factor,dataset_type): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor 
        self.dataset_type = dataset_type
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 
        # Print statements 
        #print (fname,slice)
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][slice,:,:,:]
            input_kspace  = data[self.key_kspace][slice,:,:,:]
            target = data['volfs'][slice,:,:,:]

            input_img = np.transpose(input_img,[2,0,1])
            input_kspace = np.transpose(input_kspace,[2,0,1])
            target = np.transpose(target,[2,0,1])

            return torch.from_numpy(input_img), torch.from_numpy(input_kspace), torch.from_numpy(target)
            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,acc_factor,dataset_type):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor
        self.dataset_type = dataset_type
        self.key_img = 'img_volus_{}'.format(self.acc_factor)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)


        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                #print(hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][slice,:,:,:]
            input_kspace  = data[self.key_kspace][slice,:,:,:]
            target = data['volfs'][slice,:,:,:]


            input_img = np.transpose(input_img,[2,0,1])
            input_kspace = np.transpose(input_kspace,[2,0,1])
            target = np.transpose(target,[2,0,1])

            return torch.from_numpy(input_img), torch.from_numpy(input_kspace), torch.from_numpy(target),str(fname.name),slice 
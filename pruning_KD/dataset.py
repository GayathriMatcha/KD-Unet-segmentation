import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, acc_factors,dataset_types,mask_types,train_or_valid,mask_path): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        #files = list(pathlib.Path(root).iterdir())
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        self.examples = []

        self.mask_path = mask_path.translate(translator) 
        for dataset_type in dataset_types:
            dataroot = os.path.join(root,dataset_type).translate(translator)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot,mask_type,train_or_valid).translate(translator)
                for acc_factor in acc_factors:
                    #print("acc_factor: ", acc_factor)
                    files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor)).translate(translator)).iterdir())
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        
        fname, slice,acc_factor,mask_type, dataset_type = self.examples[i] 
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
    
        with h5py.File(fname, 'r') as data:
#             print(fname)
            key_img = 'img_volus_{}'.format(acc_factor).translate(translator) 
            key_kspace = 'kspace_volus_{}'.format(acc_factor).translate(translator) 

            input_img  = data[key_img][:,:,slice]
            #print(key_img)
            input_kspace  = data[key_kspace][:,:,slice]#.astype(np.float64)
            input_kspace = npComplexToTorch(input_kspace)
    
            #target = data['volfs'][:,:,slice]
            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double
            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
            
#             if self.dataset_type == 'cardiac':
# #                 Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
#                 input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
#                 target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            #return torch.from_numpy(zf_img), torch.from_numpy(target)
#             print(os.path.join(self.mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor)))
            mask = np.load(os.path.join(self.mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor)).translate(translator))
            #print (mask)
            #acc_val = torch.Tensor([float(acc_factor[:-1].replace("_","."))])
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),torch.from_numpy(mask)

            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,acc_factor,dataset_type,mask_type,mask_path):
    #def __init__(self, root,acc_factor,dataset_type):

        # List the h5 files in root 
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        files = list(pathlib.Path(root.translate(translator)).iterdir())
        self.examples = []
        #self.acc_factor = acc_factor
        #self.dataset_type = dataset_type

        #self.key_img = 'img_volus_{}'.format(self.acc_factor)
        #self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor)

        #self.mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        self.mask_path = mask_path.translate(translator) 
        #for acc_factor in acc_factors:

        #files = list(pathlib.Path(os.path.join(root,'acc_{}'.format(acc_factor))).iterdir())
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                #acc_factor = float(acc_factor[:-1].replace("_","."))
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]

          

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice, acc_factor,mask_type, dataset_type = self.examples[i]
        # Print statements 
        #print (type(fname),slice)
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor).translate(translator) 
            key_kspace = 'kspace_volus_{}'.format(acc_factor).translate(translator) 
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]

            #kspace_cmplx = np.fft.fft2(target,norm='ortho')
            #uskspace_cmplx = kspace_cmplx * self.mask
            #zf_img = np.abs(np.fft.ifft2(uskspace_cmplx,norm='ortho'))
 
    #     if self.dataset_type == 'cardiac':
# #                 Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
#                 input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
#                 target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            # Print statements
            #print (input.shape,target.shape)
            #mask = np.load(os.path.join(self.mask_path,'mask_{}.npy'.format(acc_factor)))
            mask = np.load(os.path.join(self.mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor)).translate(translator))
            #acc_val = float(acc_factor[:-1].replace("_","."))
            #mask = os.path.join(self.mask,'{}.npy'acc_factor)
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(mask), str(fname.name),slice
            #return torch.from_numpy(zf_img), torch.from_numpy(target),str(fname.name),slice

class SliceDisplayDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root,acc_factor,dataset_type,mask_path):
    def __init__(self, root,dataset_type,mask_type,acc_factor,mask_path):
        #print (root,dataset_type,mask_type,acc_factor,mask_path)

        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        newroot = os.path.join(root,dataset_type,mask_type,'validation','acc_{}'.format(acc_factor)).translate(translator)
        # List the h5 files in root 

        files = list(pathlib.Path(newroot.translate(translator)).iterdir())
        self.examples = []
        self.acc_factor = acc_factor.translate(translator)
        self.dataset_type = dataset_type.translate(translator)

        self.key_img = 'img_volus_{}'.format(self.acc_factor).translate(translator)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor).translate(translator)

        mask_path = os.path.join(mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor)).translate(translator)
        self.mask_path = mask_path.translate(translator) 
        #self.mask = np.load(mask_path)

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                #print(hf.keys())
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]
        #print (self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i]
        # Print statements 
        #print ("inside Getitem: ",fname,slice)
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            #input_kspace  = data[self.key_kspace][:,:,slice].astype(np.float64)
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)
            mask = np.load(self.mask_path)
            #print(input_img.dtype,input_kspace.dtype,target.dtype,mask.dtype)
            #print (input.shape,target.shape)
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),mask
        

class KneeData(Dataset):

    def __init__(self, root):
        
        escapes = ''.join([chr(char) for char in range(1,32)])
        translator = str.maketrans('','',escapes)
        files = list(pathlib.Path(root.translate(translator)).iterdir())
        self.examples = []         
        for fname in sorted(files):
            self.examples.append(fname)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i] 

        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'][:])#.value
            img_und   = torch.from_numpy(data['img_und'][:])#.value
            img_und_kspace = torch.from_numpy(data['img_und_kspace'][:])#.value
            rawdata_und = torch.from_numpy(data['rawdata_und'][:])#.value
            masks = torch.from_numpy(data['masks'][:])#.value
            sensitivity = torch.from_numpy(data['sensitivity'][:])#.value
            #print("img_gt: ", img_gt.shape, "img_und: ",img_und.shape, "img_und_kspace: ",img_und_kspace.shape, "rawdata_und: ", rawdata_und.shape, "masks: ", masks.shape, "sensitivity: ",sensitivity.shape)            
            return img_gt,img_und,rawdata_und,masks,sensitivity


class KneeDataDev(Dataset):

    def __init__(self, root):

        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        files = list(pathlib.Path(root.translate(translator)).iterdir())
        self.examples = []

        for fname in sorted(files):
            self.examples.append(fname) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'][:])#.value
            img_und   = torch.from_numpy(data['img_und'][:])#.value
            img_und_kspace = torch.from_numpy(data['img_und_kspace'][:])#.value
            rawdata_und = torch.from_numpy(data['rawdata_und'][:])#.value
            masks = torch.from_numpy(data['masks'][:])#.value
            sensitivity = torch.from_numpy(data['sensitivity'][:])#.value
 
       
        return  img_gt,img_und,rawdata_und,masks,sensitivity,str(fname.name)

 

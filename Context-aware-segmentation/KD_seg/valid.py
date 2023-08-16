import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DatasetImageMaskLocaldev
# from dataset_1 import KneeDataDev
# from models import DCTeacherUNet,DCStudentUNet,UNet,SFTN#, DC#,IKD
from models2 import *
import h5py
from tqdm import tqdm
import glob

def save_reconstructions(reconstructions, out_dir):

    out_dir.mkdir(exist_ok=True)
#     print(reconstructions.keys())
#     c=reconstructions['1.h5'].shape
#     print(f'reconstructions shape is: {c}')


    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
#             print(f'recons shape is: {recons.shape}')

            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):
    
    path = args.data_path+'*.h5'
    files=glob.glob(path)
    data = DatasetImageMaskLocaldev(files,'cardiac',mode='valid')
    data_loader = DataLoader(dataset= data,batch_size=args.batch_size, num_workers=1, pin_memory=True)
   
    return data_loader


def load_model(args):

    checkpoint = torch.load(args.checkpoint)
    model_type = args.model_type
    
    if model_type == 'teacher':
        print('teacher')
        model = DCTeacherUNet()#.to(args.device)
        model = model.to(args.device)
        model.load_state_dict(checkpoint['model'])
    elif model_type == 'student':
        print('student')
        model = DCStudentUNet().to(args.device)
        model.load_state_dict(checkpoint['model'])
    elif model_type == 'UNet':
        print('UNet')
        model = UNet(num_classes=4,input_channels=1).to(args.device)
        model.load_state_dict(checkpoint['model'])
    elif model_type == 'UNet_base':
        print('UNet')
        model = UNet(n_channels=1, n_classes=4).to(args.device)
        model.load_state_dict(checkpoint['model'])
    elif model_type == 'UNet_small':
        print('UNet')
        model = UNet_S(n_channels=1, n_classes=4).to(args.device)
        model.load_state_dict(checkpoint['model'])        
    elif model_type == 'teacherSFTN':
        model = DCTeacherUNet().to(args.device)
        model_wts = model.state_dict()
        for mw in model_wts:
            for tw in checkpoint['model']:
                if tw == mw:
                    print(f'Loading weight {tw} to {mw}')
                    model_wts[mw] = checkpoint['model'][tw]
        model.load_state_dict(model_wts)

    elif model_type == 'studentSFTN':
        model = DCStudentUNet().to(args.device)
        model_wts = model.state_dict()
        for mw in model_wts:
            for tw in checkpoint['model']:
                if tw == mw:
                    print(f'Loading weight {tw} to {mw}')
                    model_wts[mw] = checkpoint['model'][tw]
        model.load_state_dict(model_wts)

    else:
        model = DCStudentUNet().to(args.device)
        model.load_state_dict(checkpoint['model'])
        print("kd")

    return model


def run_model(args, model,data_loader):

    model.eval()
    reconstructions = defaultdict(list)

    with torch.no_grad():

        for i,data in enumerate(tqdm(data_loader)):
            
            inputs,targets,fnames = data
        
            inputs   = inputs.to(args.device)
            targets  = targets.to(args.device)
#             print(f'inputs shape is {inputs.shape}')

            recons = model(inputs)[-1]
#             print (f'recons_shape1 is {recons.shape}')
            recons = recons.to('cpu').squeeze(1)
#             reconstructions.append(recons.numpy())
           
#             print (f'recons_shape2 is {recons.shape}')
#             print (recons.dtype)
            for i in range(recons.shape[0]):
                recons[i]= recons[i] 
                reconstructions[fnames[i]].append(recons[i].numpy())

    reconstructions = {
        fname:np.stack([pred for pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

#     print(reconstructions.items())
#     print(reconstructions['1.h5'])

    return reconstructions

def main(args):
    
#     print(f'args are{args.device},{args.dataset_type},{args.data-path}')
    
    data_loader = create_data_loaders(args)    
    model = load_model(args)
    reconstructions = run_model(args, model, data_loader)
    print('generation is complete')
    print(f'output directory is {args.out_dir}')
    save_reconstructions(reconstructions, args.out_dir)
    


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
#     parser.add_argument('--usmask_path',type=str,help='undersampling mask path')
    #parser.add_argument('--data_consistency',action='store_true')
    parser.add_argument('--model_type',type=str,help='model type teacher student')
    parser.add_argument('--mask_type',type=str,help='us mask path')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
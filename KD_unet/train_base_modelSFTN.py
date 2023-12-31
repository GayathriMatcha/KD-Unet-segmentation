import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset_1 import SliceData,KneeData
from trial1 import DCTeacherNet,DCStudentNet,DCTeacherNetSFTN
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm

#from utils import CriterionPairWiseforWholeFeatAfterPool, CriterionPairWiseforWholeFeatAfterPoolFeatureMaps  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

######################

def create_datasets(args):

#     train_data = SliceData(args.train_path,args.acceleration_factor,args.dataset_type)
#     dev_data = SliceData(args.validation_path,args.acceleration_factor,args.dataset_type)

#     return dev_data, train_data
    acc_factors = args.acceleration_factor.split(',')
    mask_types = args.mask_type.split(',')
    dataset_types = args.dataset_type.split(',')
    
    train_data = SliceData(args.train_path,acc_factors, dataset_types,mask_types,'train', args.usmask_path)
    dev_data = SliceData(args.validation_path,acc_factors,dataset_types,mask_types,'validation', args.usmask_path)

    return dev_data, train_data

def create_datasets_knee(args):

    train_data = KneeData(args.train_path,args.acceleration_factor,args.dataset_type)
    dev_data = KneeData(args.validation_path,args.acceleration_factor,args.dataset_type)

    return dev_data, train_data


def create_data_loaders(args):

    if args.dataset_type == 'knee':
        dev_data, train_data = create_datasets_knee(args)
    else:
        dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(2, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        #num_workers=64,
        #pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch,model,data_loader, optimizer, writer):
    
    model.train()

    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    loop = tqdm(data_loader)

    for iter, data in enumerate(loop):

        input,input_kspace,target, mask = data 
        input = input.unsqueeze(1).to(args.device)
        input_kspace = input_kspace.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)

        input = input.float()
        target = target.float()
        mask = mask.float()
        
        output = model(input,input_kspace,mask)
#         print('hduyfewly',output[0].shape)
        
 
        loss1 = F.l1_loss(output[-1],target)
    
        loss2 = F.l1_loss(output[0],target)+F.l1_loss(output[1],target)+F.l1_loss(output[2],target)+F.l1_loss(output[3],target)
        
        loss3 = F.l1_loss(output[0],output[-1])/output[-1].numel()+ F.l1_loss(output[1],output[-1])/output[-1].numel() +F.l1_loss(output[2],output[-1])/output[-1].numel()+ F.l1_loss(output[3],output[-1])/output[-1].numel()
        
#         lambda1 = 1
#         lambda2 = 1
        loss = loss1 + 1*(loss2/4) + 1*(loss3/4) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        loop.set_postfix({'Epoch': epoch, 'Loss': avg_loss})
#         break

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model,data_loader, writer):

    model.eval()

    losses = []
 
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            input,input_kspace, target, mask = data 
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
    
            input = input.float()
            target = target.float()
            mask = mask.float()
    
            output = model(input,input_kspace,mask)

#             loss = F.mse_loss(output[-1],target)
#             
            loss = F.l1_loss(output[-1],target)
            #lossSSIM = 1 - pytorch_ssim.ssim(outputT[-1],outputS[-1])
            losses.append(loss.item())
            #break
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
       
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model,data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            input,input_kspace,target,mask = data 
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)

            output = model(input.float(),input_kspace,mask)[-1]

            save_image(input, 'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target.float() - output.float()), 'Reconstruction_error')

#             break

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):

    print(args.model_type)
    if args.model_type == 'teacher':
        model = DCTeacherNet(args).to(args.device)
    if args.model_type == 'student':
        model = DCStudentNet(args).to(args.device)
    if args.model_type == 'teacherSFTN':
        model = DCTeacherNetSFTN(args).to(args.device)
    if args.model_type == 'studentSFTN':
        model = DCStudentNet(args).to(args.device)

    return model

def load_model(model,checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    model_wts = model.state_dict()

    for mw in model_wts:
        for tw in checkpoint['model']:

            if tw == mw:
                print(f'Loading weight {tw} to {mw}')
                model_wts[mw] = checkpoint['model'][tw]

    model.load_state_dict(model_wts)

    #print(checkpoint['model'])
    return model

def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    model = build_model(args)
    if args.model_type == 'studentSFTN':
        print('yes it is loading')
        model = load_model(model,args.teacher_checkpoint)
    
    #print (modeload_model(modelS,args.student_checkpoint)l)
    optimizer = build_optim(args, model.parameters())

    best_dev_loss = 1e9
    start_epoch = 0

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    
    for epoch in range(start_epoch, args.num_epochs):

        

        train_loss,train_time = train_epoch(args, epoch, model,train_loader,optimizer,writer)

        dev_loss,dev_time = evaluate(args, epoch, model, dev_loader, writer)
        
        scheduler.step()

        visualize(args, epoch, model,display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
#         scheduler.step()
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--batch-size', default=4, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=500, help='Period of loss reporting')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='us mask path')
    parser.add_argument('--model_type',type=str,help='model type') # teacher,student 
    parser.add_argument('--mask_type',type=str,help='mask type - cartesian, gaussian')
#     parser.add_argument('--teacher_checkpoint',type=str,help='teacher checkpoint')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
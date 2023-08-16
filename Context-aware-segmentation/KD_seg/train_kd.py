import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse
import glob

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from functools import reduce

from losses import LossMulti, FocalLoss 
# from models import DCStudentUNet,DCTeacherUNet,SFTN, UNet
from models2 import *
from dataset import DatasetImageMaskLocal
from kd_methods import HintLoss, Attention, Similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):
    
    train_path = args.train_path+'*.h5'
    validation_path = args.validation_path+'*.h5'

    train_file_names = glob.glob(train_path)
    random.shuffle(train_file_names)
    val_file_names = glob.glob(validation_path)
    
    train_data = DatasetImageMaskLocal(train_file_names,args.object_type,mode='train')
    dev_data = DatasetImageMaskLocal(val_file_names,args.object_type,mode='valid')
    display_data=DatasetImageMaskLocal(val_file_names,args.object_type,mode='valid')
    

    return dev_data, train_data, display_data


def create_data_loaders(args):

    dev_data, train_data, display_data = create_datasets(args)
#     display_data = [dev_data[i] for i in range(2, len(dev_data), len(dev_data) // 16)]

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
        batch_size=9,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader, display_loader

def train_epoch(args, epoch,modelT,modelS,data_loader, optimizer, writer):
    
    modelT.eval()
    modelS.train()

    avg_loss = 0.
    running_loss = 0.0
    running_loss_local = 0.0
    size = 30
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    loop = tqdm(data_loader)
    criterion = LossMulti(num_classes=4,jaccard_weight=0,device=args.device) ### CE loss
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
#     kd = HintLoss()
    kd = Attention()
#     criterion = LossMulti(num_classes=4,jaccard_weight=0,class_weights=np.array([0.1,0.5,0.2,0.2]),device=args.device) #WEIGHTEDCE
#     criterion = FocalLoss(num_classes=4,device=args.device)  #FOCAL

    for iter, data in enumerate(loop):

        inputs,targets,coord = data 
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        optimizer.zero_grad()
        
        outputT = modelT(inputs)
        
        with torch.set_grad_enabled(True):            
            
            outputS = modelS(inputs)

            outputT_feat = [outputT[0],outputT[1]]
            outputS_feat = [outputS[0],outputS[1]]
#            T=1

#             outputT_feat = [torch.sum(x**2,dim=1) for x in outputT_feat]
#             outputS_feat = [torch.sum(x**2,dim=1) for x in outputS_feat]

#             outputT_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputT_feat]
#             outputS_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputS_feat]

#             outputT_feat = [x.unsqueeze(1) for x in outputT_feat]
#             outputS_feat = [x.unsqueeze(1) for x in outputS_feat]

#            feat_l1_loss = [F.l1_loss(x,y) for x,y in zip(outputT_feat,outputS_feat)]
#            losskd = reduce((lambda x,y : x + y),feat_l1_loss)  / len(feat_l1_loss)
#             loss_kdl = kd(outputS_feat,outputT_feat)
#             loss_kd = 100000*(loss_kdl[0]+loss_kdl[1])
            loss_imitation = kl_loss(outputS[-1],outputT[-1])
#             loss_ce = criterion(outputS[-1],targets) 
#             print('losses are in order:',loss_kd,loss_imitation,loss_ce)
             
            loss = 10*criterion(outputS[-1],targets) + 0.0001*loss_imitation
#             loss = 0.01*losskd + 10*criterion(outputS[-1],targets) + 0.0001*loss_imitation
#             print('losses are in order:',loss_kd,loss_imitation,loss_ce,0.01*loss_kd,0.0001*loss_imitation,loss_ce,loss)
            

            loss.backward()
            optimizer.step()

        running_loss += loss.item()*inputs.size(0)
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {running_loss/1309:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        loop.set_postfix({'Epoch': epoch, 'Loss': running_loss/1309})
#         break

    return running_loss /1309, time.perf_counter() - start_epoch


def evaluate(args, epoch,modelT, modelS,data_loader, writer):

    modelT.eval()
    modelS.eval()

    losses = []
 
    start = time.perf_counter()
#     criterion = LossMulti(num_classes=4,jaccard_weight=0,device=args.device) ### CE loss
#     criterion = LossMulti(num_classes=4,jaccard_weight=0,device=args.device) ### CE loss
#     kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
#     kd = HintLoss()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
      
            inputs,targets = data 
            loss_local = 0.0
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            
            outputT = modelT(inputs)
            outputS = modelS(inputs)
            
#            outputT_feat = [outputT[0],outputT[1]]
#            outputS_feat = [outputS[0],outputS[1]]
            
#           feat_l1_loss = [F.l1_loss(x,y) for x,y in zip(outputT_feat,outputS_feat)]
#            losskd = reduce((lambda x,y : x + y),feat_l1_loss)  / len(feat_l1_loss)
#             loss_kd = kd(outputS[1],outputT[1])
#             loss_imitation = kl_loss(outputS[-1],outputT[-1])
             
#             loss = 0.01*loss_kd + 10*criterion(outputS[-1],targets) + 0.0001*loss_imitation
#             loss = 0.01*losskd + 5*criterion(outputS[-1],targets) + 0.0001*loss_imitation
            loss = F.nll_loss(outputS[-1], targets.squeeze(1))
            losses.append(loss.item())
#             break
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)

       
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model,data_loader, writer):
        
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25)
        writer.add_image(tag, grid, epoch)

    model.eval()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):

            inputs,targets = data 
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)[-1]
#             print('TORCH UNIQUE'+torch.unique(outputs))

            output_mask = outputs.detach().cpu().numpy() 
#             print('output mask shape',output_mask.shape)
            output_final  = np.argmax(output_mask,axis=1).astype(float)*85
#             print('output final shape',np.unique(output_final))

            output_final = torch.from_numpy(output_final).unsqueeze(1)
            
#             print('shape',output_final.shape,targets.shape)
#             print('min maxes range',inputs.min(),output_final.min(),inputs.max(),output_final.max())
            save_image(targets.float(), 'Target',9)
            save_image(output_final, 'Prediction',9)   
            

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

    modelT = DCTeacherUNet()
    modelS = DCStudentUNet()

    modelT = modelT.to(args.device)
    modelS = modelS.to(args.device)   

    return modelT,modelS

def load_model(model,checkpoint_file):
  
    #print (checkpoint_file)

    checkpoint = torch.load(checkpoint_file)
    model_wts = model.state_dict()

    for mw in model_wts:
        for tw in checkpoint['model']:
            if tw == mw:
                print(f'Loading weight {tw} to {mw}')
                model_wts[mw] = checkpoint['model'][tw]

    model.load_state_dict(model_wts)
    return model

def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr)
#     optimizer = Adam(model.parameters(), lr=1e-4)
    return optimizer


def main(args):
    
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    modelT,modelS = build_model(args)

    modelT = load_model(modelT,args.teacher_checkpoint)
    print(f'teacher wts loaded')
    modelS = load_model(modelS,args.student_checkpoint)

    optimizer = build_optim(args, modelS.parameters())

    best_dev_loss = 1e9
    start_epoch = 0
    
    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):

        train_loss,train_time = train_epoch(args, epoch, modelT,modelS,train_loader,optimizer,writer)

        dev_loss,dev_time = evaluate(args, epoch, modelT, modelS, dev_loader, writer)
#         scheduler.step()
        visualize(args, epoch, modelS,display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, modelS, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--batch-size', default=16, type=int,  help='Mini batch size')
    parser.add_argument('--valbatch-size', default=9, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
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

#     parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--object_type',type=str,help='cardiac,kirby')
    parser.add_argument('--model_type',type=str,help='model type') # teacher,student 
    parser.add_argument('--teacher_checkpoint',type=str,help='teacher checkpoint')
    parser.add_argument('--student_checkpoint',type=str,help='student checkpoint')
    parser.add_argument('--use_pretrained',action='store_true',help='for selecting whether to use student_pretrained_not')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)

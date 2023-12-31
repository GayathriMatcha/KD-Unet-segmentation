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
from dataset import SliceData,KneeData
from models import DCTeacherNet #,DCStudentNet
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from functools import reduce
import pytorch_msssim
import torch.nn.utils.prune as prune
# import pytorch_ssim

# from utils import CriterionPairWiseforWholeFeatAfterPool, CriterionPairWiseforWholeFeatAfterPoolFeatureMaps  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):                                 ##########change3

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



    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )

#     train_loader_error = DataLoader(
#         dataset=train_data,
#     )

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
    return train_loader, dev_loader, display_loader         #,train_loader_error

####################################################################################################################################
def train_epoch(args, epoch,modelT,modelS,data_loader, optimizer, writer):#,error_range):# , vgg):
    
    modelT.eval() 
    modelS.train()

    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    alpha = 0.5
    ssim_factor = 0.1 

#     criterian=SSIMLoss()

    loop = tqdm(data_loader)                   ##############change4
    for iter, data in enumerate(loop):

        input,input_kspace,target,mask = data # Return kspace also we can ignore that for train and test 
        input = input.unsqueeze(1).to(args.device)
        input_kspace = input_kspace.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)      ##############change5

        input = input.float()
        target = target.float()
        mask = mask.float()

        outputT = modelT(input,input_kspace,mask)
        outputS = modelS(input,input_kspace,mask)
        
        lossSG = F.l1_loss(outputS[-1],target) # ground truth loss 
        lossST = F.l1_loss(outputS[-1],outputT[-1]) / outputT[-1].numel() # student - teacher loss 
  
        
        '''
        outputT_feat = [outputT[0][2],outputT[1][2],outputT[2][2],outputT[3][2],outputT[4][2]]
        outputS_feat = [outputS[0][1],outputS[1][1],outputS[2][1],outputS[3][1],outputS[4][1]]

        outputT_feat = [torch.sum(x**2,dim=1) for x in outputT_feat]
        outputS_feat = [torch.sum(x**2,dim=1) for x in outputS_feat]
        outputT_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputT_feat]
        outputS_feat = [torch.nn.functional.normalize(x,dim=0) for x in outputS_feat]

        outputT_feat = [x.unsqueeze(1) for x in outputT_feat]
        outputS_feat = [x.unsqueeze(1) for x in outputS_feat]
#         print(outputT_feat.shape,outputT_feat.shape)

        feat_l1_loss = [F.l1_loss(x,y)/y.numel() for x,y in zip(outputT_feat,outputS_feat)]      
#         feat_ssim_loss = [1 - pytorch_ssim.ssim(x,y).item() for x,y in zip(outputT_feat,outputS_feat)]
        feat_ssim_loss = [1 - pytorch_msssim.ssim(x,y) for x,y in zip(outputT_feat,outputS_feat)]

#         print (feat_l1_loss,feat_ssim_loss)
        loss_l1   = reduce((lambda x,y : x + y),feat_l1_loss)  / len(feat_l1_loss) 
        loss_ssim = reduce((lambda x,y : x + y),feat_ssim_loss)  / len(feat_ssim_loss) 
        
        loss = loss_l1 + ssim_factor * loss_ssim
        
         '''

        #print (loss_l1,loss_ssim)
        #lossT = feat_loss_sum
        #print (lossG,lossT)
        #loss = alpha * lossSG + ( 1 - alpha) * error_scaleT * lossST 

        
        if args.imitation_required:
#             print('true its happening')
            loss = lossSG + lossST 
        else:
            loss = lossSG
        #loss = torch.min(lossG,lossT)


        optimizer.zero_grad()
 
        loss.backward()
        zero_pruned_gradients(args,modelS)
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
        #break
    if epoch % 100 == 0:
        get_pruning_statistics(modelS, verbose=True)
    return avg_loss, time.perf_counter() - start_epoch
####################################################################################################################################
    

def evaluate(args, epoch, modelT,modelS,data_loader, writer):

    modelT.eval()
    modelS.eval()

    #losses_mse   = []
    losses = []
    #losses_ssim  = []
 
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            input,input_kspace, target, mask = data # Return kspace also we can ignore that for train and test
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
    
            input = input.float()
            target = target.float()
            mask = mask.float()
    
            outputS = modelS(input,input_kspace,mask)


#             loss = F.mse_loss(outputS[-1],target)
            loss = F.l1_loss(outputS[-1],target)

            losses.append(loss.item())
                    
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
        
    return np.mean(losses), time.perf_counter() - start

####################################################################################################################################

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

            break
####################################################################################################################################
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

####################################################################################################################################
def build_model(args):
 
    modelT = DCTeacherNet().to(args.device)
    modelS = DCTeacherNet().to(args.device)
#     parameters_to_prune = (
# #         (modelS.tcascade1.conv1, 'weight'),
# #         (modelS.tcascade1.conv2, 'weight'),
# #         (modelS.tcascade1.conv3, 'weight'),
# #         (modelS.tcascade1.conv4, 'weight'),
# #         (modelS.tcascade1.conv5, 'weight'),
# #         (modelS.tcascade2.conv1, 'weight'),
# #         (modelS.tcascade2.conv2, 'weight'),
# #         (modelS.tcascade2.conv3, 'weight'),
# #         (modelS.tcascade2.conv4, 'weight'),
# #         (modelS.tcascade2.conv5, 'weight'),
# #         (modelS.tcascade3.conv1, 'weight'),
# #         (modelS.tcascade3.conv2, 'weight'),
# #         (modelS.tcascade3.conv3, 'weight'),
# #         (modelS.tcascade3.conv4, 'weight'),
# #         (modelS.tcascade3.conv5, 'weight'),
# #         (modelS.tcascade4.conv1, 'weight'),
# #         (modelS.tcascade4.conv2, 'weight'),
# #         (modelS.tcascade4.conv3, 'weight'),
# #         (modelS.tcascade4.conv4, 'weight'),
# #         (modelS.tcascade4.conv5, 'weight'),
# #         (modelS.tcascade5.conv1, 'weight'),
# #         (modelS.tcascade5.conv2, 'weight'),
# #         (modelS.tcascade5.conv3, 'weight'),
#         (modelS.tcascade5.conv4, 'weight'),
#         (modelS.tcascade5.conv5, 'weight'),
#     )
#     prune.global_unstructured(
#         parameters_to_prune,
#         pruning_method=prune.L1Unstructured,
#         amount=float(args.sparsity),
#     )
#     for module, thing in parameters_to_prune:
# #         print(module)
#         prune.remove(module,'weight')

    return modelT,modelS

def load_model(model,checkpoint_file):
  
    #print (checkpoint_file)

    checkpoint = torch.load(checkpoint_file)    
    model.load_state_dict(checkpoint['model'])
#     model_wts = model.state_dict()

# #     for (tw,mw) in zip(checkpoint['model'],model_wts):
#     for mw in model_wts:
# #         print(f' model weights {mw}')s
#         for tw in checkpoint['model']:
# #         tmp = mw.split('.')
# #             print(f'teacher weights {tw} and model weights {mw}')

#             if tw == mw:
#                 print(f'Loading weight {tw} to {mw}')
#                 model_wts[mw] = checkpoint['model'][tw]

#     model.load_state_dict(model_wts)

    #print(checkpoint['model'])
    return model

####################################################################################################################################
def get_error_range_teacher(loader,model):

    losses = []
    print ("Finding max and min error between teacher and target")

    for data in tqdm(loader):

        input,input_kspace, target,mask = data # Return kspace also we can ignore that for train and test
        
        input  = input.unsqueeze(1).float().to(args.device)
        input_kspace = input_kspace.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).float().to(args.device)

        output = model(input,input_kspace,mask)[-1]
        
        loss = F.l1_loss(output,target).detach().cpu().numpy()

        losses.append(loss)

    min_error,max_error = np.min(losses),np.max(losses)
    #pdb.set_trace()

    return max_error #- min_error 

####################################################################################################################################
def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer
def zero_pruned_gradients(args,model):
    """
    Function used for zeroing gradients of pruned weights
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            param_data = param.data.cpu().numpy()
            param_grad = param.grad.data.cpu().numpy()
            param_grad = np.where(param_data == 0.0, 0, param_grad)
            param.grad.data = torch.from_numpy(param_grad).to(args.device)
            
def get_pruning_statistics(model, verbose=True):
#     model = copy.deepcopy(self.model)
#     if model_path:
#         model.load_state_dict(torch.load(model_path))
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        if verbose:
            print(
                f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
            )
    if verbose:
        print(
            f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
        )
    return round((nonzero / total) * 100, 1)  
####################################################################################################################################
def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    modelT,modelS = build_model(args)

    modelT = load_model(modelT,args.teacher_checkpoint)
#     args.student_pretrained=True
#     args.imitation_required=True
    
#     if not args.student_pretrained:
    if args.student_pretrained:
        print("dtysjgsadfukyasd")
        modelS = load_model(modelS,args.student_checkpoint)

    optimizer = build_optim(args, modelS.parameters())

    best_dev_loss = 1e9
    start_epoch = 0

    train_loader, dev_loader, display_loader = create_data_loaders(args)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    
    for epoch in range(start_epoch, args.num_epochs):

        train_loss,train_time = train_epoch(args, epoch, modelT,modelS,train_loader,optimizer,writer)
        dev_loss,dev_time = evaluate(args, epoch, modelT, modelS, dev_loader, writer)
        
        scheduler.step()

        visualize(args, epoch, modelS,display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, modelS, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()

####################################################################################################################################
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

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='us mask path')
    parser.add_argument('--teacher_checkpoint',type=str,help='teacher checkpoint')
    parser.add_argument('--student_checkpoint',type=str,help='student checkpoint')
    parser.add_argument('--student_pretrained',action='store_true',help='for selecting whether to use student_pretrained_not')
    parser.add_argument('--imitation_required',action='store_true',help='option to select imitation loss')
    parser.add_argument('--mask_type',type=str,help='us mask path')
    parser.add_argument('--sparsity',type=str,help='factor with which 0.9,0.95,0.975') 
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
####################################################################################################################################

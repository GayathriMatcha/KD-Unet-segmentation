import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from torch.optim import Adam
from tqdm import tqdm
import logging
from torch import nn
import numpy as np
import h5py
import torchvision
import random
from tensorboardX import SummaryWriter

from utils import visualize,evaluate
from losses import LossMulti, FocalLoss 
from models import SUNet
from dataset import DatasetImageMaskLocal

if __name__ == "__main__":

    train_path  =  '/media/hticimg/data1/Data/MRI/cardiac_mri_acdc_dataset/train/*.h5'
    val_path  =  '/media/hticimg/data1/Data/MRI/cardiac_mri_acdc_dataset/test/*.h5'
    object_type = 'cardiac'
    model_type = 'SUNet'
    save_path = '/home/hticimg/gayathri/Context-aware-segmentation/KD_seg/{}_{}/models_local'.format(object_type,model_type)

    use_pretrained = False
    pretrained_model_path = 'PRETRAINEDPATH'
   
    #TODO:Add hyperparams to ArgParse. 
    batch_size = 16
    val_batch_size = 9 
    no_of_epochs = 150

    size = 30 #ROI Size -> 60x60.
    cuda_no = 0
    CUDA_SELECT = "cuda:{}".format(cuda_no)

    writer = SummaryWriter(log_dir='/home/hticimg/gayathri/Context-aware-segmentation/KD_seg/{}_{}/models_local/summary'.format(object_type,model_type))

    logging.basicConfig(filename="log_{}_run_student_local.txt".format(object_type),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M',
                            level=logging.INFO)

    logging.info('Model: SUNet + Loss:  {}'.format(object_type)) 

    train_file_names = glob.glob(train_path)
    random.shuffle(train_file_names)

    val_file_names = glob.glob(val_path)

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    model = SUNet(num_classes=4,input_channels=1)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    print(model)

    # To handle epoch start number and pretrained weight 
    epoch_start = '0'
    if(use_pretrained):
        print("Loading Model {}".format(os.path.basename(pretrained_model_path)))
        model.load_state_dict(torch.load(pretrained_model_path))
        epoch_start = os.path.basename(pretrained_model_path).split('.')[0]
        print(epoch_start)

    
    trainLoader   = DataLoader(DatasetImageMaskLocal(train_file_names,object_type,mode='train'),batch_size=batch_size)
    devLoader     = DataLoader(DatasetImageMaskLocal(val_file_names,object_type,mode='valid'))
    displayLoader = DataLoader(DatasetImageMaskLocal(val_file_names,object_type,mode='valid'),batch_size=val_batch_size)

    optimizer = Adam(model.parameters(), lr=1e-4)

    #TODO:Include LossType in Argparse.
    criterion_global = LossMulti(num_classes=4,jaccard_weight=0,device=device) 
    criterion_local = LossMulti(num_classes=4,jaccard_weight=0,device=device) 


    for epoch in tqdm(range(int(epoch_start)+1,int(epoch_start)+1+no_of_epochs)):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0
        running_loss_local = 0.0

        for i,(inputs,targets,coord) in enumerate(tqdm(trainLoader)):

            loss_local = 0.0
            
            model.train()
            inputs   = inputs.to(device)
            targets  = targets.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss_global = criterion_global(outputs,targets)
                
                local_pred_ = []
                local_gt_ = []
                for i,coord in enumerate(coord):
                        
                    cx,cy = int(coord[0].item()),int(coord[1].item())
                    local_pred = outputs[i,:,cy-size:cy+size,cx-size:cx+size]
                    local_gt = targets[i,:,cy-size:cy+size,cx-size:cx+size]
                    local_pred_.append(local_pred)
                    local_gt_.append(local_gt)         
                loss_local = criterion_local(torch.stack(local_pred_),torch.stack(local_gt_))
                
                loss = loss_global + loss_local
                writer.add_scalar('loss_global', loss_global, epoch)
                writer.add_scalar('loss_local', loss_local, epoch)
                writer.add_scalar('Total_Loss', loss.item(), epoch)
                
                loss.backward()
                optimizer.step()

            running_loss += loss.item()*inputs.size(0)
            running_loss_local += loss_local*inputs.size(0)
        epoch_loss = running_loss / len(train_file_names)
        epoch_loss_local = running_loss_local / len(train_file_names)

        if epoch%1 == 0:
            dev_loss,dev_time = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar('loss_valid', dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, val_batch_size)
            print("Epoch:{} Global Loss:{} Local Loss:{} Dev Loss:{}".format(epoch,epoch_loss,epoch_loss_local,dev_loss))
        
        else:
            print("Epoch:{} Global Loss:{} Local Loss:{}".format(epoch,epoch_loss,epoch_loss_local))
        
        logging.info('epoch:{} train_loss:{} train_loss_local:{}'.format(epoch,epoch_loss,epoch_loss_local))
        if epoch%15==0:
            torch.save(model.state_dict(),os.path.join(save_path,str(epoch)+'.pt'))

import os
from os import listdir
import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim
from skimage.filters import laplace
from tqdm import tqdm
from os.path import isfile, join
import numpy 
import cv2

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch import Tensor


# from medpy import metric

# def dice_coef(inputs, targets):
#     y_inputs = inputs.flatten()
#     y_targets = targets.flatten()

#     intersection = (y_inputs * y_targets).sum()
# #     smooth = 0.0001
#     smooth = 1.0
#     dice = (2. * intersection + smooth) / (y_inputs.sum() + y_targets.sum() + smooth)
# #     print(f'all sizes inp,trg,A_sum,B_sum are {y_inputs.shape},{y_targets.shape}')
#     return dice

# def dice_coef_multilabel(targets,outputs):
#     dice=0
#     numLabels=4
# #     for index in range(numLabels):
# #         dice += dice_coef(y_true[index,:,:], y_pred[index,:,:])
#      # taking average
# #     print(targets.shape,outputs.shape)
# #     eps = 0.0001
#     smooth = 1
#     eps=1e-8
#     targets = targets.squeeze(1)
#     for cls in range(numLabels): 
#         dice_target = (targets == cls)#.float()
#         dice_output = np.exp(outputs[:,cls]) 
        
#         print('target and output shape is:', dice_output.shape, dice_target.shape)
#         intersection = (dice_output * dice_target).sum()

#         union = dice_output.sum() + dice_target.sum()
#         dice +=1-(2.*intersection + smooth + eps) / (union + smooth + eps)
     
#     return dice/numLabels

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(target, input, multiclass: bool = True):
    # Dice loss (objective to minimize) between 0 and 1
    input = torch.from_numpy(input)
#     target = torch.from_numpy(target)
    target = target.squeeze(1)
    target=torch.Tensor(target).long()
#     target_one_hot = torch.eye(4)[target]
#     target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
#     print(f'unique data is {torch.unique(target)}')
#     print(f'target before one hot {target.shape}')
    input = F.softmax(input, dim=1).float()
    target = F.one_hot(target,4).permute(0, 3, 1, 2).float()
#     print(f'target before one hot {target.shape}')
    
#     dice_loss(F.softmax(outputS[-1], dim=1).float(),F.one_hot(targets.squeeze(1),4).permute(0, 3, 1, 2).float(), multiclass=True)
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def jaccordian_multilabel(targets,outputs):
    loss=0.0
    eps = 1e-7
    num_classes =4
    targets=torch.from_numpy(targets)
    outputs=torch.from_numpy(outputs)
    targets = targets.squeeze(1)
    for cls in range(num_classes): 
        jaccard_target = (targets == cls)#.float()
        jaccard_output = outputs[:, cls]#.exp() 
#         print('jaccard shapes ',jaccard_target.shape,jaccard_output.shape)
        intersection = (jaccard_output * jaccard_target).sum()

        union = jaccard_output.sum() + jaccard_target.sum()
#         loss -= torch.log((intersection + eps) / (union - intersection + eps))
        loss += 1-((intersection + eps) / (union - intersection + eps))

    return loss/num_classes
#     return loss

def focalloss_multilabel(targets,outputs): 
    alpha=4.0
    gamma=2.0
    eps=1e-7
    num_classes=4
    loss=0.0
    targets = targets.squeeze(1)
    target_one_hot = torch.eye(num_classes)[targets]
    target_one_hot = target_one_hot.permute(2,0,1).float()
    outputs=torch.from_numpy(outputs)
    cls_prob = outputs#.exp() + eps #Counter LogSoftmax with exp to get softmax.
    
    ce = target_one_hot * -1 * torch.log(cls_prob)
    weight = target_one_hot * ((1-cls_prob) ** gamma)
    fl = alpha * weight * ce
    reduced_fl,_ = torch.max(fl,dim=1)
    loss = torch.mean(reduced_fl)

    return loss

def crossentropy_multilabel(targets,outputs):
#     loss=0.0
#     loss = torch.nn.NLLLoss()
    criterion = nn.CrossEntropyLoss() 
    targets = torch.from_numpy(targets)
    outputs = torch.from_numpy(outputs)
    ce = criterion(outputs, targets.squeeze(1).long())
    return ce
    
METRIC_FUNCS = dict(
    DICE_Loss = dice_loss,
#     FOCAL_Loss = focalloss_multilabel,
    JACCORDIAN_Loss = jaccordian_multilabel,
    CE_loss =crossentropy_multilabel
)
class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }


    '''
    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )
    '''

    def get_report(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
#             f'{name} = {means[name]:.4g}' for name in metric_names
        )




def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in tqdm(args.target_path.iterdir()):
        print (tgt_file)
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
            
            target = target[recons_key]#[:]
            target = np.pad(target,pad_width=((5,5),(5,5)),mode='constant')
            target = target.astype(np.uint8)
            target = np.expand_dims(target, 0)
            target = np.expand_dims(target, 0)
#             print(f'target shape is {target.shape}')
            
#             target_one_hot = torch.eye(4)[target]
#             target_one_hot = target_one_hot.permute(2,0,1).float() 
            
#             eps=1e-7
            recons = recons['reconstruction'][:]
#             print(f'recons shape is {recons.shape}')
# #             recons= recons
#             cls_prob = np.exp(recons) + eps #Counter LogSoftmax with exp to get softmax.
            
#             cls_prob =cls_prob.reshape((cls_prob.shape[1],cls_prob.shape[2],cls_prob.shape[3]))
            
        
            
# #             print(f'recons shape is {recons.shape},{np.unique(recons)}')
# #             print(f'cs_prob shape is {cls_prob.shape},{np.unique(cls_prob)}')
# #             print(f'target one hot shape is {target_one_hot.shape},{torch.unique(target_one_hot)}')
    
# #             print(np.unique(recons),np.unique(cls_prob))
#             cls_prob=torch.from_numpy(cls_prob)
#             cls_prob=torch.round(cls_prob)
# #             print(f'cs_prob shape is {cls_prob.shape},{torch.unique(cls_prob)}')
         
#             metrics.push(target_one_hot, cls_prob)
            metrics.push(target, recons)
            
    return metrics



if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')

    args = parser.parse_args()

    recons_key = 'mask'
    metrics = evaluate(args, recons_key)
    metrics_report = metrics.get_report()

    with open(args.report_path / 'report.txt','w') as f:
        f.write(metrics_report)

    #print(metrics)

# import os
# from os import listdir
# import argparse
# import pathlib
# from argparse import ArgumentParser

# import h5py
# import numpy as np
# from runstats import Statistics
# from skimage.measure import compare_psnr, compare_ssim
# from skimage.filters import laplace
# from tqdm import tqdm
# from os.path import isfile, join
# import numpy 
# import cv2
# import torch
# from torch.nn import functional as F

# # from medpy import metric

# def dice_coef(inputs, targets):
#     y_inputs = inputs.flatten()
#     y_targets = targets.flatten()

#     intersection = (y_inputs * y_targets).sum()
# #     smooth = 0.0001
#     smooth = 1.0
#     dice = (2. * intersection + smooth) / (y_inputs.sum() + y_targets.sum() + smooth)
# #     print(f'all sizes inp,trg,A_sum,B_sum are {y_inputs.shape},{y_targets.shape}')
#     return dice

# # class DiceLoss(nn.Module):
# #     def __init__(self, weight=None, size_average=True):
# #         super(DiceLoss, self).__init__()

# #     def forward(self, inputs, targets, smooth=1):
        
# #         inputs = F.sigmoid(inputs)       
# #         inputs = inputs.view(-1)
# #         targets = targets.view(-1)       
# #         intersection = (inputs * targets).sum()                            
# #         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
# #         return 1 - dice

# def dice_coef_multilabel(targets,outputs):
#     dice=0
#     numLabels=4
# #     for index in range(numLabels):
# #         dice += dice_coef(y_true[index,:,:], y_pred[index,:,:])
#      # taking average
# #     print(targets.shape,outputs.shape)
# #     eps = 0.0001
#     smooth = 1
#     eps=1e-8
#     targets = targets.squeeze(1)
#     for cls in range(numLabels): 
#         dice_target = (targets == cls)#.float()
#         dice_output = np.exp(outputs[:,cls]) 
        
#         print('target and output shape is:', dice_output.shape, dice_target.shape)
#         intersection = (dice_output * dice_target).sum()

#         union = dice_output.sum() + dice_target.sum()
#         dice +=1-(2.*intersection + smooth + eps) / (union + smooth + eps)
     
#     return dice/numLabels


# def jaccordian_multilabel(targets,outputs):
#     loss=0.0
#     eps = 1e-7
#     num_classes =4
#     targets=torch.from_numpy(targets)
#     outputs=torch.from_numpy(outputs)
#     targets = targets.squeeze(1)
#     for cls in range(num_classes): 
#         jaccard_target = (targets == cls)#.float()
#         jaccard_output = outputs[:, cls].exp() 
# #         print('jaccard shapes ',jaccard_target.shape,jaccard_output.shape)
#         intersection = (jaccard_output * jaccard_target).sum()

#         union = jaccard_output.sum() + jaccard_target.sum()
# #         loss -= torch.log((intersection + eps) / (union - intersection + eps))
#         loss += 1-((intersection + eps) / (union - intersection + eps))

#     return loss/num_classes
# #     return loss

# # def focalloss_multilabel(targets,outputs): 
# #     alpha=4.0
# #     gamma=2.0
# #     eps=1e-7
# #     num_classes=4
# #     loss=0.0
# ###     targets=torch.from_numpy(targets)
# #     targets = targets.squeeze(1)
# # #     print(targets.shape)
# #     target_one_hot = torch.eye(num_classes)[targets]
# #     target_one_hot = target_one_hot.permute(2,0,1).float()
# #     outputs=torch.from_numpy(outputs)
# #     cls_prob = outputs.exp() + eps #Counter LogSoftmax with exp to get softmax.
    
# #     ce = target_one_hot * -1 * torch.log(cls_prob)
# #     weight = target_one_hot * ((1-cls_prob) ** gamma)
# #     fl = alpha * weight * ce
# #     reduced_fl,_ = torch.max(fl,dim=1)
# #     loss = torch.mean(reduced_fl)

# #     return loss

# def crossentropy_multilabel(targets,outputs):
# #     loss=0.0
#     loss = torch.nn.NLLLoss()
#     targets = torch.from_numpy(targets)#.long()
#     outputs = torch.from_numpy(outputs)#.long()
# #     targets = targets.type(torch.LongTensor)
# #     outputs = outputs.type(torch.LongTensor)
# #     print(outputs.shape,targets.shape)
#     ce = loss(outputs, targets.squeeze(1).long())
#     return ce
    
# METRIC_FUNCS = dict(
#     CE_loss =crossentropy_multilabel,
#     DC_Loss = dice_coef_multilabel,
# #     FOCAL_Loss = focalloss_multilabel,
#     JACCORDIAN_Loss = jaccordian_multilabel
    
# )
# class Metrics:
#     """
#     Maintains running statistics for a given collection of metrics.
#     """

#     def __init__(self, metric_funcs):
#         self.metrics = {
#             metric: Statistics() for metric in metric_funcs
#         }

#     def push(self, target, recons):
#         for metric, func in METRIC_FUNCS.items():
#             self.metrics[metric].push(func(target, recons))

#     def means(self):
#         return {
#             metric: stat.mean() for metric, stat in self.metrics.items()
#         }

#     def stddevs(self):
#         return {
#             metric: stat.stddev() for metric, stat in self.metrics.items()
#         }


#     '''
#     def __repr__(self):
#         means = self.means()
#         stddevs = self.stddevs()
#         metric_names = sorted(list(means))
#         return ' '.join(
#             f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
#         )
#     '''

#     def get_report(self):
#         means = self.means()
#         stddevs = self.stddevs()
#         metric_names = sorted(list(means))
#         return ' '.join(
#             f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
# #             f'{name} = {means[name]:.4g}' for name in metric_names
#         )




# def evaluate(args, recons_key):
#     metrics = Metrics(METRIC_FUNCS)

#     for tgt_file in tqdm(args.target_path.iterdir()):
#         print (tgt_file)
#         with h5py.File(tgt_file) as target, h5py.File(
#           args.predictions_path / tgt_file.name) as recons:
            
#             target = target[recons_key]#[:]
#             target = np.pad(target,pad_width=((5,5),(5,5)),mode='constant')
#             target = target.astype(np.uint8)
#             target = np.expand_dims(target, 0)
#             target = np.expand_dims(target, 0)
#             print(target.shape)
            
# #             target_one_hot = torch.eye(4)[target]
# #             target_one_hot = target_one_hot.permute(2,0,1).float() 
            
# #             eps=1e-7
#             recons = recons['reconstruction'][:]
# # #             recons= recons
# #             cls_prob = np.exp(recons) + eps #Counter LogSoftmax with exp to get softmax.
            
# #             cls_prob =cls_prob.reshape((cls_prob.shape[1],cls_prob.shape[2],cls_prob.shape[3]))
            
        
            
# # #             print(f'recons shape is {recons.shape},{np.unique(recons)}')
# # #             print(f'cs_prob shape is {cls_prob.shape},{np.unique(cls_prob)}')
# # #             print(f'target one hot shape is {target_one_hot.shape},{torch.unique(target_one_hot)}')
    
# # #             print(np.unique(recons),np.unique(cls_prob))
# #             cls_prob=torch.from_numpy(cls_prob)
# #             cls_prob=torch.round(cls_prob)
# # #             print(f'cs_prob shape is {cls_prob.shape},{torch.unique(cls_prob)}')
         
# #             metrics.push(target_one_hot, cls_prob)
#             metrics.push(target, recons)
            
#     return metrics



# if __name__ == '__main__':
#     parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--target-path', type=pathlib.Path, required=True,
#                         help='Path to the ground truth data')
#     parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
#                         help='Path to reconstructions')
#     parser.add_argument('--report-path', type=pathlib.Path, required=True,
#                         help='Path to save metrics')

#     args = parser.parse_args()

#     recons_key = 'mask'
#     metrics = evaluate(args, recons_key)
#     metrics_report = metrics.get_report()

#     with open(args.report_path / 'report.txt','w') as f:
#         f.write(metrics_report)

#     #print(metrics)

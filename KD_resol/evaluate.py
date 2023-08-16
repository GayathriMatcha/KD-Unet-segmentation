import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import laplace
from tqdm import tqdm
from tqdm import tqdm
from utils import complex_abs
import torch
# from sewar.full_ref import vifp

# adding hfn metric 
def hfn(gt,pred):

    hfn_total = []

    for ii in range(gt.shape[-1]):
        gt_slice = gt[:,:,ii]
        pred_slice = pred[:,:,ii]

        pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
        pred_slice[pred_slice>1] = 1

        gt_slice_laplace = laplace(gt_slice)        
        pred_slice_laplace = laplace(pred_slice)

        hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
        hfn_total.append(hfn_slice)

    return np.mean(hfn_total)


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    #return compare_ssim(
    #    gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    #)
    return structural_similarity(gt,pred,multichannel=True, data_range=gt.max())

METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfn
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
        )




def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in tqdm(args.target_path.iterdir()):
        #print (tgt_file)
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
            
#             target = target[recons_key][:]            
#             target = torch.from_numpy(target) #[110,256,256,2]
# #             target = target.permute(0,3,1,2) #[110,2,256,256]
#             print (f'target shape is {target.shape}')
            
#             recons = recons['reconstruction'][:] #[110,2,256,256]
#             recons = torch.from_numpy(recons)  #[110,2,256,256]
#             recons = recons.permute(0,2,3,1)
                    
#             recons_abs = complex_abs(recons)#.unsqueeze(1)
#             target_abs = complex_abs(target).unsqueeze(1)
# #             target = torch.from_numpy(target)
# #             print (target.shape)
# # #             target = low_or_high_freq_recover(target,target)
# #             target = complex_abs(target).numpy()
# #             recons = recons['reconstruction'][:]
# #             recons = complex_abs(recons).numpy()
#             print (f'target shape and recon shape is {target.shape},{recons.shape}')
            target = target[recons_key][:]
            target = torch.from_numpy(target)
#             target = low_or_high_freq_recover(target,target)
            target = complex_abs(target).numpy()
            recons = recons['reconstruction'][:]
#             print (f'target shape and recon shape is {target.shape},{recons.shape}')
            recons = np.transpose(recons,[1,2,0])
            target = np.transpose(target,[1,2,0])
#             print (target.shape,recons.shape) 

            if len(target.shape) == 2:
                target = np.expand_dims(target,2) # added for knee mri 
            #print (target.shape,recons.shape)
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

    recons_key = 'volfs'
    metrics = evaluate(args, recons_key)
    metrics_report = metrics.get_report()

    with open(args.report_path / 'report.txt','w') as f:
        f.write(metrics_report)

    #print(metrics)
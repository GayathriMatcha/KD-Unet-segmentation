import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import math

    
class DataConsistencyLayer(nn.Module):

    def __init__(self):

        super(DataConsistencyLayer,self).__init__()

    def forward(self,predicted_img,us_kspace,us_mask):

        kspace_predicted_img = torch.fft.fft2(predicted_img,norm = "ortho")

        us_kspace_complex = us_kspace[:,:,:,:,0]+us_kspace[:,:,:,:,1]*1j

        updated_kspace1  = us_mask * us_kspace_complex

        updated_kspace2  = (1 - us_mask) * kspace_predicted_img

        updated_kspace = updated_kspace1 + updated_kspace2

        updated_img  = torch.fft.ifft2(updated_kspace,norm = "ortho")

        updated_img = torch.view_as_real(updated_img)
        
        update_img_abs = updated_img[:,:,:,:,0] # taking real part only, change done on Sep 18 '19 bcos taking abs till bring in the distortion due to imag part also. this was verified was done by simple experiment on FFT, mask and IFFT


        return update_img_abs.float()


class TeacherNet(nn.Module):
    
    def __init__(self):
        super(TeacherNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = self.conv5(x4)
        
        return x1,x2,x3,x4,x5
#         return x5

##############################################################################################################DC-UNET
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)
        output_latent = output

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)

class DC_UNET(nn.Module):
    def __init__(self,args,n_ch=1,nc=5):
        super(DC_UNET,self).__init__()
        self.nc = nc
        #conv_blocks = []
        unet = []
        dcs = []
        for i in range(nc):
            #cnnblock = five_layerCNN_MAML(args,n_ch)
            #conv_blocks.append(cnnblock)   
            unet.append(UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0))
            dcs.append(DataConsistencyLayer())

        #self.conv_blocks = nn.ModuleList(conv_blocks)
        self.unet = nn.ModuleList(unet)
        self.dcs = nn.ModuleList(dcs)


    def forward(self, us_query_input,ksp_query_imgs,ksp_mask_query):
        x = us_query_input
        y=[]
        for i in range(self.nc):
#             print(f'one: {x.shape}')
            #x_cnn = self.conv_blocks[i](x)
            #x = x_cnn+x
            x_u = self.unet[i](x)
            x = x_u+x
            x = self.dcs[i](x, ksp_query_imgs, ksp_mask_query)
#             y.append(x)
#         print(len(y))
        return x    

######################################################################3333 TEACHER 333###############################################
class DCTeacherNet(nn.Module):

    def __init__(self,args):

        super(DCTeacherNet,self).__init__()
        n_ch = 1

        self.tcascade1 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        self.tcascade2 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        self.tcascade3 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        self.tcascade4 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        self.tcascade5 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        


        self.tdc = DataConsistencyLayer()

    def forward(self,x,x_k,us_mask):

        x1 = self.tcascade1(x)  
        x1_dc = self.tdc(x1+x, x_k,us_mask)

        x2 = self.tcascade2(x1_dc)
        x2_dc = self.tdc(x2+x1_dc, x_k,us_mask)

        x3 = self.tcascade3(x2_dc)
        x3_dc = self.tdc(x3+x2_dc, x_k,us_mask)

        x4 = self.tcascade4(x3_dc)
        x4_dc = self.tdc(x4+x3_dc, x_k,us_mask)

        x5 = self.tcascade5(x4_dc)
        x5_dc = self.tdc(x5+x4_dc, x_k,us_mask)

        return x1,x2,x3,x4,x5,x5_dc

########################################################################333 STUDENT #######################################################
class StudentNet(nn.Module):
    
    def __init__(self):
        super(StudentNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)
        
        return x1,x2,x3
        # return x3


class DCStudentNet(nn.Module):

    def __init__(self,args):

        super(DCStudentNet,self).__init__()

        self.scascade1 = StudentNet()
        self.scascade2 = StudentNet()
        self.scascade3 = StudentNet()
        self.scascade4 = StudentNet()
        self.scascade5 = StudentNet()

        self.sdc = DataConsistencyLayer()#sus_mask

    def forward(self,x,x_k,us_mask):

        x1 = self.scascade1(x) # list of channel outputs 
        x1_dc = self.sdc(x1[-1],x_k,us_mask)

        x2 = self.scascade2(x1_dc)
        x2_dc = self.sdc(x2[-1],x_k,us_mask)

        x3 = self.scascade3(x2_dc)
        x3_dc = self.sdc(x3[-1],x_k,us_mask)

        x4 = self.scascade4(x3_dc)
        x4_dc = self.sdc(x4[-1],x_k,us_mask)

        x5 = self.scascade5(x4_dc)
        x5_dc = self.sdc(x5[-1],x_k,us_mask)

        return x1,x2,x3,x4,x5,x5_dc

############################################################*SFTN_TEACHER*#

class DCTeacherNetSFTN(nn.Module):

    def __init__(self,args):

        super(DCTeacherNetSFTN,self).__init__()
        n_ch = 1
    
        self.tcascade1 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        self.tcascade2 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        self.tcascade3 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        self.tcascade4 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        self.tcascade5 = UnetModel(in_chans=n_ch,out_chans=n_ch,chans=32,num_pool_layers=3,drop_prob=0.0)
        
        self.scascade1 = StudentNet()
        self.scascade2 = StudentNet()
        self.scascade3 = StudentNet()
        self.scascade4 = StudentNet()
        self.scascade5 = StudentNet()

        self.tdc = DataConsistencyLayer()
        self.sdc = DataConsistencyLayer()

    def forward(self,x,x_k,us_mask):
        if self.training:

            x1 = self.tcascade1(x)  
            x1_dc = self.tdc(x1+x, x_k,us_mask)

            x2 = self.tcascade2(x1_dc)
            x2_dc = self.tdc(x2+x1_dc, x_k,us_mask)

            x3 = self.tcascade3(x2_dc)
            x3_dc = self.tdc(x3+x2_dc, x_k,us_mask)

            x4 = self.tcascade4(x3_dc)
            x4_dc = self.tdc(x4+x3_dc, x_k,us_mask)

            x5 = self.tcascade5(x4_dc)
            x5_dc = self.tdc(x5+x4_dc, x_k,us_mask)

    ##################################################
            x_s1 = self.scascade2(x1_dc)
            x_s1dc = self.sdc(x_s1[-1],x_k,us_mask)

            x_s1 = self.scascade3(x_s1dc)
            x_s1dc = self.sdc(x_s1[-1],x_k,us_mask)

            x_s1 = self.scascade4(x_s1dc)
            x_s1dc = self.sdc(x_s1[-1],x_k,us_mask)

            x_s1 = self.scascade5(x_s1dc)
            x_s1dc = self.sdc(x_s1[-1],x_k,us_mask)

    ####################################################        
            x_s2 = self.scascade3(x2_dc)
            x_s2dc = self.sdc(x_s2[-1],x_k,us_mask)

            x_s2 = self.scascade4(x_s2dc)
            x_s2dc = self.sdc(x_s2[-1],x_k,us_mask)

            x_s2 = self.scascade5(x_s2dc)
            x_s2dc = self.sdc(x_s2[-1],x_k,us_mask)

    #######################################################

            x_s3 = self.scascade4(x3_dc)
            x_s3dc = self.sdc(x_s3[-1],x_k,us_mask)

            x_s3 = self.scascade5(x_s3dc)
            x_s3dc = self.sdc(x_s3[-1],x_k,us_mask)

    #########################################################       

            x_s4 = self.scascade5(x4_dc)
            x_s4dc = self.sdc(x_s4[-1],x_k,us_mask)
            
            op = x_s1dc, x_s2dc, x_s3dc, x_s4dc, x5_dc
            
        else: #Eval mode
            
            x1 = self.tcascade1(x) # list of channel outputs 
            x1_dc = self.tdc(x1+x, x_k,us_mask)

            x2 = self.tcascade2(x1_dc)
            x2_dc = self.tdc(x2+x1_dc, x_k,us_mask)

            x3 = self.tcascade3(x2_dc)
            x3_dc = self.tdc(x3+x2_dc, x_k,us_mask)

            x4 = self.tcascade4(x3_dc)
            x4_dc = self.tdc(x4+x3_dc, x_k,us_mask)

            x5 = self.tcascade5(x4_dc)
            x5_dc = self.tdc(x5+x4_dc, x_k,us_mask)

            op = x1,x2,x3,x4,x5,x5_dc
            
        
        return op
        
############################################################*INTERACTIVE_KD*#
    
class IKD(nn.Module):
    def __init__(self) -> None:
        super(IKD,self).__init__()
        
        self.tcascades = nn.ModuleList([TeacherNet() for _ in range(5)])

        for item in self.tcascades:
            for p in item.parameters():
                p.requires_grad = False

        self.scascades = nn.ModuleList([StudentNet() for _ in range(5)])

        self.dc = DataConsistencyLayer()

    def forward(self,x,x_k,us_mask,cfg=[True]*5):
        
        if self.training: # Training mode
#             print(f'cfg = {cfg}')

            if cfg[0]:
                x = self.scascades[0](x)

            else:
                x = self.tcascades[0](x)

#             x = self.scascades[0](x)
            x = self.dc(x[-1],x_k,us_mask)

            if cfg[1]: # Select student
                x = self.scascades[1](x)

            else:
                x = self.tcascades[1](x)

            x = self.dc(x[-1],x_k,us_mask)


            if cfg[2]: # Select student
                x = self.scascades[2](x)

            else:
                x = self.tcascades[2](x)

            x = self.dc(x[-1],x_k,us_mask)


            if cfg[3]: # Select student
                x = self.scascades[3](x)

            else:
                x = self.tcascades[3](x)

            x = self.dc(x[-1],x_k,us_mask)

            if cfg[4]: # Select student
                x = self.scascades[4](x)

            else:
                x = self.tcascades[4](x)

            x = self.dc(x[-1],x_k,us_mask)


        else: # Eval mode
            x = self.scascades[0](x)  
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[1](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[2](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[3](x)
            x = self.dc(x[-1],x_k,us_mask)

            x = self.scascades[4](x)
            x = self.dc(x[-1],x_k,us_mask)

        return x
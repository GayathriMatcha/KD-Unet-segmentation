import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import math
import transforms as T

##############################################################################################################################

class DataConsistencyLayer(nn.Module):

    def __init__(self, noise_lvl=None):
        super(DataConsistencyLayer, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))
    
    def forward(self, x, k0, mask, sensitivity,shape):

        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
#         print('1:',x.shape,sensitivity.shape)

        x = x.permute(0, 2, 3, 1)
        x = T.complex_img_pad(x,shape) ##Have to edit

        x = T.complex_multiply(x[...,0].unsqueeze(1), x[...,1].unsqueeze(1), 
                               sensitivity[...,0], sensitivity[...,1])

        k = T.dc_fft2(x)
              
        v = self.noise_lvl
        if v is not None: 
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0) 
        else:  
            out = (1 - mask) * k + mask * k0
    
        # ### backward op ### #
        
        x = T.dc_ifft2(out)
#         print("x: ",x.shape, "out: ", out.shape, "Sens: ", sensitivity.shape)      
        Sx = T.complex_multiply(x[...,0], x[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)     
        
#         SS = T.complex_multiply(sensitivity[...,0], 
#                                 sensitivity[...,1], 
#                                 sensitivity[...,0], 
#                                -sensitivity[...,1]).sum(dim=1)
        #print("Sx: ", Sx.shape) T.complex_center_crop(x,(320,320))
        Sx = T.complex_center_crop(Sx,(320,320))
        Sx = Sx.permute(0, 3, 1, 2)
        return Sx#, SS
    
##############################################################################################################################
    
class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def forward(self, cnn, Sx):
        
        x = self.para*cnn + (1 - self.para)*Sx
#         print("sx: ", Sx.shape, "cnn: ", cnn.shape, "x: ", x.shape)
        return x
##############################################################################################################################
class StudentNet(nn.Module):
    
    def __init__(self):
        super(StudentNet, self).__init__()
        
        self.conv1 = nn.Conv2d(2,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,2,kernel_size=3,stride=1,padding=1)
        
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

        self.sdc = DataConsistencyLayer(0.1)
        self.swa = weightedAverageTerm(0.1)

    def forward(self,x,k, m, s,shape ):
        
        x_crop = T.complex_center_crop(x,(320,320))
        x_crop = x_crop.permute(0, 3, 1, 2)

        x1 = self.scascade1(x_crop)      
        x1_dc = self.sdc(x1[-1],k, m, s,shape)
#         print(x_crop.shape,x1[-1].shape,x1_dc.shape)
        x1_wa = self.swa(x_crop + x1[-1], x1_dc)
        

        x2 = self.scascade2(x1_wa)
        x2_dc = self.sdc(x2[-1],k, m, s,shape)
        x2_wa = self.swa(x_crop + x2[-1], x2_dc)

        x3 = self.scascade3(x2_wa)
        x3_dc = self.sdc(x3[-1],k, m, s,shape)
        x3_wa = self.swa(x_crop + x3[-1], x3_dc)

        x4 = self.scascade4(x3_wa)
        x4_dc = self.sdc(x4[-1],k, m, s,shape)
        x4_wa = self.swa(x_crop + x4[-1], x4_dc)

        x5 = self.scascade5(x4_wa)
        x5_dc = self.sdc(x5[-1],k, m, s,shape)
        x5_wa = self.swa(x_crop + x5[-1], x5_dc)

        return x1,x2,x3,x4,x5,x5_wa
    
########################################################################TeacherNet#######################################################
class TeacherNet(nn.Module):
    
    def __init__(self):
        super(TeacherNet, self).__init__()
        
        self.conv1 = nn.Conv2d(2,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(32,2,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = self.conv5(x4)
        
        return x1,x2,x3,x4,x5
#         return x5


class DCTeacherNet(nn.Module):

    def __init__(self,args):

        super(DCTeacherNet,self).__init__()

        self.tcascade1 = TeacherNet()
        self.tcascade2 = TeacherNet()
        self.tcascade3 = TeacherNet()
        self.tcascade4 = TeacherNet()
        self.tcascade5 = TeacherNet()


        self.tdc = DataConsistencyLayer(0.1)
        self.twa = weightedAverageTerm(0.1)

    def forward(self,x,k, m, s,shape):

        x_crop = T.complex_center_crop(x,(320,320))
        x_crop = x_crop.permute(0, 3, 1, 2)

        x1 = self.tcascade1(x_crop)
        x1_dc = self.tdc(x1[-1],k, m, s,shape)
        x1_wa = self.twa(x_crop + x1[-1], x1_dc)

        x2 = self.tcascade2(x1_wa)
        x2_dc = self.tdc(x2[-1],k, m, s,shape)
        x2_wa = self.twa(x_crop + x2[-1], x2_dc)
        
        x3 = self.tcascade2(x2_wa)
        x3_dc = self.tdc(x3[-1],k, m, s,shape)
        x3_wa = self.twa(x_crop + x3[-1], x3_dc)
        
        x4 = self.tcascade2(x3_wa)
        x4_dc = self.tdc(x2[-1],k, m, s,shape)
        x4_wa = self.twa(x_crop + x4[-1], x4_dc)
        
        x5 = self.tcascade2(x4_wa)
        x5_dc = self.tdc(x2[-1],k, m, s,shape)
        x1_wa = self.twa(x_crop + x5[-1], x5_dc)

        return x1,x2,x3,x4,x5,x5_dc

############################################################*SFTN_TEACHER*###################################

class DCTeacherNetSFTN(nn.Module):

    def __init__(self,args):

        super(DCTeacherNetSFTN,self).__init__()
        

        self.tcascade1 = TeacherNet()
        self.tcascade2 = TeacherNet()
        self.tcascade3 = TeacherNet()
        self.tcascade4 = TeacherNet()
        self.tcascade5 = TeacherNet()
        
        self.scascade1 = StudentNet()
        self.scascade2 = StudentNet()
        self.scascade3 = StudentNet()
        self.scascade4 = StudentNet()
        self.scascade5 = StudentNet()

        self.tdc = DataConsistencyLayer(0.1)
        self.sdc = DataConsistencyLayer(0.1)
        self.wa = weightedAverageTerm(0.1)

    def forward(self,x,k, m, s, shape):
        if self.training:
            
            x_crop = T.complex_center_crop(x,(320,320))
            x_crop = x_crop.permute(0, 3, 1, 2)
            
            x1 = self.tcascade1(x_crop)
            x1_dc = self.tdc(x1[-1],k, m, s,shape)
            x1_wa = self.wa(x_crop + x1[-1], x1_dc)

            x2 = self.tcascade2(x1_dc)
            x2_dc = self.tdc(x2[-1],k, m, s,shape)
            x2_wa = self.wa(x_crop + x2[-1], x2_dc)

            x3 = self.tcascade2(x2_dc)
            x3_dc = self.tdc(x3[-1],k, m, s,shape)
            x3_wa = self.wa(x_crop + x3[-1], x3_dc)

            x4 = self.tcascade2(x3_dc)
            x4_dc = self.tdc(x2[-1],k, m, s,shape)
            x4_wa = self.wa(x_crop + x4[-1], x4_dc)

            x5 = self.tcascade2(x4_dc)
            x5_dc = self.tdc(x2[-1],k, m, s,shape)
            x1_wa = self.wa(x_crop + x5[-1], x5_dc)

    ##################################################
            x_s1 = self.scascade2(x1_wa)
            x_s1dc = self.sdc(x_s1[-1],k, m, s,shape)
            x_s1wa = self.wa(x_crop + x_s1[-1], x_s1dc)

            x_s1 = self.scascade3(x_s1wa)
            x_s1dc = self.sdc(x_s1[-1],k, m, s,shape)
            x_s1wa = self.wa(x_crop + x_s1[-1], x_s1dc)

            x_s1 = self.scascade4(x_s1wa)
            x_s1dc = self.sdc(x_s1[-1],k, m, s,shape)
            x_s1wa = self.wa(x_crop + x_s1[-1], x_s1dc)

            x_s1 = self.scascade5(x_s1wa)
            x_s1dc = self.sdc(x_s1[-1],k, m, s,shape)
            x_s1wa = self.wa(x_crop + x_s1[-1], x_s1dc)

    ####################################################        
            x_s2 = self.scascade3(x2_wa)
            x_s2dc = self.sdc(x_s2[-1],k, m, s,shape)
            x_s2wa = self.wa(x_crop + x_s2[-1], x_s2dc)

            x_s2 = self.scascade4(x_s2wa)
            x_s2dc = self.sdc(x_s2[-1],k, m, s,shape)
            x_s2wa = self.wa(x_crop + x_s2[-1], x_s2dc)

            x_s2 = self.scascade5(x_s2wa)
            x_s2dc = self.sdc(x_s2[-1],k, m, s,shape)
            x_s2wa = self.wa(x_crop + x_s2[-1], x_s2dc)

    #######################################################

            x_s3 = self.scascade4(x3_wa)
            x_s3dc = self.sdc(x_s3[-1],k, m, s,shape)
            x_s3wa = self.wa(x_crop + x_s3[-1], x_s3dc)

            x_s3 = self.scascade5(x_s3wa)
            x_s3dc = self.sdc(x_s3[-1],k, m, s,shape)
            x_s3wa = self.wa(x_crop + x_s3[-1], x_s3dc)

    #########################################################       

            x_s4 = self.scascade5(x4_wa)
            x_s4dc = self.sdc(x_s4[-1],k, m, s,shape)
            x_s4wa = self.wa(x_crop + x_s4[-1], x_s4dc)
            
            op = x_s1wa, x_s2wa, x_s3wa, x_s4wa, x5_wa
            
        else: #Eval mode
            
            x_crop = T.complex_center_crop(x,(320,320))
            x_crop = x_crop.permute(0, 3, 1, 2)
            
            x1 = self.tcascade1(x_crop)
            x1_dc = self.tdc(x1[-1],k, m, s,shape)
            x1_wa = self.twa(x_crop + x1[-1], x1_dc)

            x2 = self.tcascade2(x1_wa)
            x2_dc = self.tdc(x2[-1],k, m, s,shape)
            x2_wa = self.twa(x_crop + x2[-1], x2_dc)

            x3 = self.tcascade2(x2_wa)
            x3_dc = self.tdc(x3[-1],k, m, s,shape)
            x3_wa = self.twa(x_crop + x3[-1], x3_dc)

            x4 = self.tcascade2(x3_wa)
            x4_dc = self.tdc(x2[-1],k, m, s,shape)
            x4_wa = self.twa(x_crop + x4[-1], x4_dc)

            x5 = self.tcascade2(x4_wa)
            x5_dc = self.tdc(x2[-1],k, m, s,shape)
            x1_wa = self.twa(x_crop + x5[-1], x5_dc)

            op = x1,x2,x3,x4,x5,x5_wa
            
        
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

    def forward(self,x,k, m, s ,cfg=[True]*5):
        
        if self.training: 

            if cfg[0]:
                x = self.scascades[0](x)

            else:
                x = self.tcascades[0](x)

            x = self.dc(x[-1],k, m, s )

            if cfg[1]: 
                x = self.scascades[1](x)

            else:
                x = self.tcascades[1](x)

            x = self.dc(x[-1],k, m, s )


            if cfg[2]:
                x = self.scascades[2](x)

            else:
                x = self.tcascades[2](x)

            x = self.dc(x[-1],k, m, s )


            if cfg[3]: 
                x = self.scascades[3](x)

            else:
                x = self.tcascades[3](x)

            x = self.dc(x[-1],k, m, s )

            if cfg[4]:
                x = self.scascades[4](x)

            else:
                x = self.tcascades[4](x)

            x = self.dc(x[-1],k, m, s )


        else: 
            x = self.scascades[0](x)  
            x = self.dc(x[-1],k, m, s )

            x = self.scascades[1](x)
            x = self.dc(x[-1],k, m, s )

            x = self.scascades[2](x)
            x = self.dc(x[-1],k, m, s )

            x = self.scascades[3](x)
            x = self.dc(x[-1],k, m, s )

            x = self.scascades[4](x)
            x = self.dc(x[-1],k, m, s )

        return x
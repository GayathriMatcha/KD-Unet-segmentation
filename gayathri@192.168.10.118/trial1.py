import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import math
import transforms as T

class DataConsistencyLayer(nn.Module):

    def __init__(self, noise_lvl=None):
        super(DataConsistencyLayer, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))
    
    def forward(self, x, k0, mask, sensitivity):

        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
#         print('1:',x.shape,sensitivity.shape)
        x = x.permute(0, 2, 3, 1)
#         x = T.complex_img_pad(x,shape) ##Have to edit
        x = T.complex_multiply(x[...,0].unsqueeze(1), x[...,1].unsqueeze(1), 
                               sensitivity[...,0], sensitivity[...,1])
     
#         print('2:',x.shape)

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
        #print("Sx: ", Sx.shape) 
        Sx = Sx.permute(0, 3, 1, 2)
        return Sx#, SS


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


        self.tdc = DataConsistencyLayer()

    def forward(self,x,k, m, s):
#         print(f'input shape : 1 {x.shape}')

        x_crop = T.complex_center_crop(x,(320,320))
        shape = x.shape


        x1 = self.tcascade1(x.permute(0, 3, 1, 2))
        x1_dc = self.tdc(x1[-1],k, m, s)

        x2 = self.tcascade2(x1_dc)
        x2_dc = self.tdc(x2[-1],k, m, s)

        x3 = self.tcascade3(x2_dc)
        x3_dc = self.tdc(x3[-1],k, m, s )

        x4 = self.tcascade4(x3_dc)
        x4_dc = self.tdc(x4[-1],k, m, s )

        x5 = self.tcascade5(x4_dc)
        x5_dc = self.tdc(x5[-1] ,k, m, s )

        return x1,x2,x3,x4,x5,x5_dc


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

    def forward(self,x,k, m, s ):

        x1 = self.scascade1(x.permute(0, 3, 1, 2))      
        x1_dc = self.sdc(x1[-1],k, m, s )

        x2 = self.scascade2(x1_dc)
        x2_dc = self.sdc(x2[-1],k, m, s )

        x3 = self.scascade3(x2_dc)
        x3_dc = self.sdc(x3[-1],k, m, s )

        x4 = self.scascade4(x3_dc)
        x4_dc = self.sdc(x4[-1],k, m, s )

        x5 = self.scascade5(x4_dc)
        x5_dc = self.sdc(x5[-1],k, m, s )

        return x1,x2,x3,x4,x5,x5_dc

############################################################*SFTN_TEACHER*#

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

        self.tdc = DataConsistencyLayer()
        self.sdc = DataConsistencyLayer()

    def forward(self,x,k, m, s ):
        if self.training:
            
            x1 = self.tcascade1(x.permute(0, 3, 1, 2))  
            x1_dc = self.tdc(x1[-1],k, m, s )

            x2 = self.tcascade2(x1_dc)
            x2_dc = self.tdc(x2[-1],k, m, s )

            x3 = self.tcascade3(x2_dc)
            x3_dc = self.tdc(x3[-1],k, m, s )

            x4 = self.tcascade4(x3_dc)
            x4_dc = self.tdc(x4[-1],k, m, s )

            x5 = self.tcascade5(x4_dc)
            x5_dc = self.tdc(x5[-1],k, m, s )

    ##################################################
            x_s1 = self.scascade2(x1_dc)
            x_s1dc = self.sdc(x_s1[-1],k, m, s )

            x_s1 = self.scascade3(x_s1dc)
            x_s1dc = self.sdc(x_s1[-1],k, m, s )

            x_s1 = self.scascade4(x_s1dc)
            x_s1dc = self.sdc(x_s1[-1],k, m, s )

            x_s1 = self.scascade5(x_s1dc)
            x_s1dc = self.sdc(x_s1[-1],k, m, s )

    ####################################################        
            x_s2 = self.scascade3(x2_dc)
            x_s2dc = self.sdc(x_s2[-1],k, m, s )

            x_s2 = self.scascade4(x_s2dc)
            x_s2dc = self.sdc(x_s2[-1],k, m, s )

            x_s2 = self.scascade5(x_s2dc)
            x_s2dc = self.sdc(x_s2[-1],k, m, s )

    #######################################################

            x_s3 = self.scascade4(x3_dc)
            x_s3dc = self.sdc(x_s3[-1],k, m, s )

            x_s3 = self.scascade5(x_s3dc)
            x_s3dc = self.sdc(x_s3[-1],k, m, s )

    #########################################################       

            x_s4 = self.scascade5(x4_dc)
            x_s4dc = self.sdc(x_s4[-1],k, m, s )
            
            op = x_s1dc, x_s2dc, x_s3dc, x_s4dc, x5_dc
            
        else: #Eval mode
            
            x1 = self.tcascade1(x) 
            x1_dc = self.tdc(x1[-1],k, m, s )

            x2 = self.tcascade2(x1_dc)
            x2_dc = self.tdc(x2[-1],k, m, s )

            x3 = self.tcascade3(x2_dc)
            x3_dc = self.tdc(x3[-1],k, m, s )

            x4 = self.tcascade4(x3_dc)
            x4_dc = self.tdc(x4[-1],k, m, s )

            x5 = self.tcascade5(x4_dc)
            x5_dc = self.tdc(x5[-1],k, m, s )
            
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

    def forward(self,x,k, m, s ,cfg=[True]*5):
        
        if self.training: # Training mode
#             print(f'cfg = {cfg}')

            if cfg[0]:
                x = self.scascades[0](x)

            else:
                x = self.tcascades[0](x)

#             x = self.scascades[0](x)
            x = self.dc(x[-1],k, m, s )

            if cfg[1]: # Select student
                x = self.scascades[1](x)

            else:
                x = self.tcascades[1](x)

            x = self.dc(x[-1],k, m, s )


            if cfg[2]: # Select student
                x = self.scascades[2](x)

            else:
                x = self.tcascades[2](x)

            x = self.dc(x[-1],k, m, s )


            if cfg[3]: # Select student
                x = self.scascades[3](x)

            else:
                x = self.tcascades[3](x)

            x = self.dc(x[-1],k, m, s )

            if cfg[4]: # Select student
                x = self.scascades[4](x)

            else:
                x = self.tcascades[4](x)

            x = self.dc(x[-1],k, m, s )


        else: # Eval mode
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
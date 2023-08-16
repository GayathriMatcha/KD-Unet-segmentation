import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import math


#### Super-resolution vdsr 
########################################################### Teacher ###################################################################    

class TeacherVDSR(nn.Module):
    
    def __init__(self):

        super(TeacherVDSR, self).__init__()
        
        self.tconv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)
        self.tconv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv6 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv7 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv9 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv10 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv11 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv12 = nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        
        x1 = F.relu(self.tconv1(x))
        x2 = F.relu(self.tconv2(x1))
        x3 = F.relu(self.tconv3(x2))
        x4 = F.relu(self.tconv4(x3))
        x5 = F.relu(self.tconv5(x4))
        x6 = F.relu(self.tconv6(x5))
        x7 = F.relu(self.tconv7(x6))
        x8 = F.relu(self.tconv8(x7))
        x9 = F.relu(self.tconv9(x8))
        x10 = F.relu(self.tconv10(x9))
        x11 = F.relu(self.tconv11(x10))
        x12 = self.tconv12(x11)
        x12 = x + x12

        return x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
########################################################### Student ###################################################################    

class StudentVDSR(nn.Module):
    
    def __init__(self):

        super(StudentVDSR, self).__init__()
        
        self.sconv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)
        self.sconv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv4 = nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
       
    def forward(self, x):

        x1 = F.relu(self.sconv1(x))
        x2 = F.relu(self.sconv2(x1))
        x3 = F.relu(self.sconv3(x2))
        x4 = self.sconv4(x3)

        x4 = x4 + x 

        return x1,x2,x3,x4
########################################################## SFTN ###################################################################    
class SFTN(nn.Module):
    
    def __init__(self):

        super(SFTN, self).__init__()
    
    
        self.tconv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)
        self.tconv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv6 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv7 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv9 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv10 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv11 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv12 = nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        
        self.sconv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)
        self.sconv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv4 = nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        if self.training:
            x1 = F.relu(self.tconv1(x))
            x2 = F.relu(self.tconv2(x1))
            x3 = F.relu(self.tconv3(x2))
            x4 = F.relu(self.tconv4(x3))
            x5 = F.relu(self.tconv5(x4))
            x6 = F.relu(self.tconv6(x5))
            x7 = F.relu(self.tconv7(x6))
            x8 = F.relu(self.tconv8(x7))
            x9 = F.relu(self.tconv9(x8))
            x10 = F.relu(self.tconv10(x9))
            x11 = F.relu(self.tconv11(x10))
            x12 = self.tconv12(x11)
            x12 = x + x12 

            xs1_2 = F.relu(self.sconv2(x3))
            xs1_3 = F.relu(self.sconv3(xs1_2))
            xs1_4 = self.sconv4(xs1_3)
            xs1_4 = xs1_4 + x 

            xs2_3 = F.relu(self.sconv3(x6))
            xs2_4 = self.sconv4(xs2_3)
            xs2_4 = xs2_4 + x

            xs3_4 = self.sconv4(x9)
            xs3_4 = xs3_4 + x 

            op = xs1_4,xs2_4,xs3_4,x12
            
        else: #Eval mode
            
            x1 = F.relu(self.tconv1(x))
            x2 = F.relu(self.tconv2(x1))
            x3 = F.relu(self.tconv3(x2))
            x4 = F.relu(self.tconv4(x3))
            x5 = F.relu(self.tconv5(x4))
            x6 = F.relu(self.tconv6(x5))
            x7 = F.relu(self.tconv7(x6))
            x8 = F.relu(self.tconv8(x7))
            x9 = F.relu(self.tconv9(x8))
            x10 = F.relu(self.tconv10(x9))
            x11 = F.relu(self.tconv11(x10))
            x12 = self.tconv12(x11)
            x12 = x + x12
            
            op = x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
            
        
        return op
########################################################## SFTN_2###################################################################    
'''
class SFTN2(nn.Module):
    
    def __init__(self):

        super(SFTN2, self).__init__()
    
    
        self.tconv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)
        self.tconv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv6 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv7 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv9 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv10 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv11 = nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        
        self.sconv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)
        self.sconv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv6 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv7 = nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        if self.training:
            x1 = F.relu(self.tconv1(x))
            x2 = F.relu(self.tconv2(x1))
            x3 = F.relu(self.tconv3(x2))
            x4 = F.relu(self.tconv4(x3))
            x5 = F.relu(self.tconv5(x4))
            x6 = F.relu(self.tconv6(x5))
            x7 = F.relu(self.tconv7(x6))
            x8 = F.relu(self.tconv8(x7))
            x9 = F.relu(self.tconv9(x8))
            x10 = F.relu(self.tconv10(x9))
            x11 = self.tconv11(x10)
            x11 = x + x11 
            
            xs1_3 = F.relu(self.sconv3(x3))
            xs1_4 = F.relu(self.sconv4(xs1_3))
            xs1_5 = F.relu(self.sconv5(xs1_4))
            xs1_6 = F.relu(self.sconv6(xs1_5))
            xs1_7 = self.sconv7(xs1_6)
            xs1_7 = xs1_7 + x 
            
            xs2_5 = F.relu(self.sconv5(x6))
            xs2_6 = F.relu(self.sconv6(xs2_5))
            xs2_7 = self.sconv7(xs2_6)
            xs2_7 = xs2_7 + x 

            xs3_7 = self.sconv7(x9)
            xs3_7 = xs3_7 + x 

            op = xs1_7,xs2_7,xs3_7,x11
            
        else: #Eval mode
            
            x1 = F.relu(self.tconv1(x))
            x2 = F.relu(self.tconv2(x1))
            x3 = F.relu(self.tconv3(x2))
            x4 = F.relu(self.tconv4(x3))
            x5 = F.relu(self.tconv5(x4))
            x6 = F.relu(self.tconv6(x5))
            x7 = F.relu(self.tconv7(x6))
            x8 = F.relu(self.tconv8(x7))
            x9 = F.relu(self.tconv9(x8))
            x10 = F.relu(self.tconv10(x9))
            x11 = self.tconv11(x10)
            x11 = x + x11
            
            op = x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11            
        
        return op 
########################################################## SFTN_30000000###################################################################        
class SFTN3(nn.Module):
    
    def __init__(self):

        super(SFTN3, self).__init__()
    
    
        self.tconv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)
        self.tconv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv6 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv7 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv9 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv10 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.tconv11 = nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        
        self.sconv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1)
        self.sconv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv6 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.sconv7 = nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        if self.training:
            x1 = F.relu(self.tconv1(x))
            x2 = F.relu(self.tconv2(x1))
            x3 = F.relu(self.tconv3(x2))
            x4 = F.relu(self.tconv4(x3))
            x5 = F.relu(self.tconv5(x4))
            x6 = F.relu(self.tconv6(x5))
            x7 = F.relu(self.tconv7(x6))
            x8 = F.relu(self.tconv8(x7))
            x9 = F.relu(self.tconv9(x8))
            x10 = F.relu(self.tconv10(x9))
            x11 = self.tconv11(x10)
            x11 = x + x11 
            
            xs1_2 = F.relu(self.sconv3(x2))
            xs1_3 = F.relu(self.sconv3(xs1_2))
            xs1_4 = F.relu(self.sconv4(xs1_3))
            xs1_5 = F.relu(self.sconv5(xs1_4))
            xs1_6 = F.relu(self.sconv6(xs1_5))
            xs1_7 = self.sconv7(xs1_6)
            xs1_7 = xs1_7 + x 
            
            xs2_4 = F.relu(self.sconv3(x5))
            xs2_5 = F.relu(self.sconv5(xs2_4))
            xs2_6 = F.relu(self.sconv6(xs2_5))
            xs2_7 = self.sconv7(xs2_6)
            xs2_7 = xs2_7 + x 

            xs3_6 = self.sconv6(x8)
            xs3_7 = self.sconv7(xs3_6)
            xs3_7 = xs3_7 + x 

            op = xs1_7,xs2_7,xs3_7,x11
            
        else: #Eval mode
            
            x1 = F.relu(self.tconv1(x))
            x2 = F.relu(self.tconv2(x1))
            x3 = F.relu(self.tconv3(x2))
            x4 = F.relu(self.tconv4(x3))
            x5 = F.relu(self.tconv5(x4))
            x6 = F.relu(self.tconv6(x5))
            x7 = F.relu(self.tconv7(x6))
            x8 = F.relu(self.tconv8(x7))
            x9 = F.relu(self.tconv9(x8))
            x10 = F.relu(self.tconv10(x9))
            x11 = self.tconv11(x10)
            x11 = x + x11
            
            op = x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11
            
        return op
'''
#######################################################################################################
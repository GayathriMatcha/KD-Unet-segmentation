from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np


def conv3x3(in_, out, padding=1):
    return nn.Conv2d(in_, out, 3, padding=padding)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, padding=1, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out, padding)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x

class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int, padding=1):
        super().__init__()
        self.l1 = Conv3BN(in_, out, padding)
        self.l2 = Conv3BN(out, out, padding)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
#################################################### BASE UNET ##############################################################
class UNet(nn.Module):
    
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels = 3,
                 filters_base: int = 32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=(1, 2, 4, 8, 16),
                 bottom_s=4,
                 num_classes=1,
                 padding=1,
                 add_output=True):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0], padding))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf, padding))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i], padding))
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, padding)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))

        if self.add_output:
            x_out = self.conv_final(x_out)
            if self.num_classes > 1:
                x_out = F.log_softmax(x_out, dim=1)
                
        return x_out
#####################################################################################################################################
##########################################################   UNET MODIFIED for cascaded ################################################
# class UNet(nn.Module):
    
#     output_downscaled = 1
#     module = UNetModule

#     def __init__(self,
#                  input_channels = 3,
#                  filters_base: int = 32,
#                  down_filter_factors=(1, 2, 4, 8, 16),
#                  up_filter_factors=(1, 2, 4, 8, 16),
#                  bottom_s=4,
#                  num_classes=1,
#                  padding=1,
#                  add_output=True):
#         super().__init__()
#         self.num_classes = num_classes
#         assert len(down_filter_factors) == len(up_filter_factors)
#         assert down_filter_factors[-1] == up_filter_factors[-1]
#         down_filter_sizes = [filters_base * s for s in down_filter_factors]
#         up_filter_sizes = [filters_base * s for s in up_filter_factors]
#         self.down, self.up = nn.ModuleList(), nn.ModuleList()
#         self.down.append(self.module(input_channels, down_filter_sizes[0], padding))
#         for prev_i, nf in enumerate(down_filter_sizes[1:]):
#             self.down.append(self.module(down_filter_sizes[prev_i], nf, padding))
#         for prev_i, nf in enumerate(up_filter_sizes[1:]):
#             self.up.append(self.module(
#                 down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i], padding))
#         pool = nn.MaxPool2d(2, 2)
#         pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
#         upsample = nn.Upsample(scale_factor=2)
#         upsample_bottom = nn.Upsample(scale_factor=bottom_s)
#         self.downsamplers = [None] + [pool] * (len(self.down) - 1)
#         self.downsamplers[-1] = pool_bottom
#         self.upsamplers = [upsample] * len(self.up)
#         self.upsamplers[-1] = upsample_bottom
#         self.add_output = add_output
# #         if add_output:
#         self.conv_final_1 = nn.Conv2d(up_filter_sizes[0], 1, padding)
#         self.conv_final_2 = nn.Conv2d(up_filter_sizes[0], num_classes, padding)

#     def forward(self, x):
#         xs = []
#         for downsample, down in zip(self.downsamplers, self.down):
#             x_in = x if downsample is None else downsample(xs[-1])
#             x_out = down(x_in)
#             xs.append(x_out)

#         x_out = xs[-1]
#         for x_skip, upsample, up in reversed(
#                 list(zip(xs[:-1], self.upsamplers, self.up))):
#             x_out = upsample(x_out)
#             x_out = up(torch.cat([x_out, x_skip], 1))
            
#         if self.add_output:
#             x_out = self.conv_final_2(x_out)
#             if self.num_classes > 1:
#                 x_out = F.log_softmax(x_out, dim=1)
#         else:
#             x_out = self.conv_final_1(x_out)
                
#         return x_out

########################################### CASCADED Teacher UNET with depth 5 #######################################################

class DCTeacherUNet(nn.Module):

    def __init__(self):

        super(DCTeacherUNet,self).__init__()

        self.tcascade1 = UNet(num_classes=4,input_channels=1,add_output=False)
        self.tcascade2 = UNet(num_classes=4,input_channels=1,add_output=False)
        self.tcascade3 = UNet(num_classes=4,input_channels=1,add_output= True)

    def forward(self,x):
        
        x1 = self.tcascade1(x)
        x2 = self.tcascade2(x1)
        x3 = self.tcascade3(x2)


        return x1,x2,x3
########################################### CASCADED Student UNET with depth 3 #######################################################
   
class DCStudentUNet(nn.Module):

    def __init__(self):

        super(DCStudentUNet,self).__init__()

        self.scascade1 = UNet(num_classes=4,input_channels=1,down_filter_factors=(1, 2, 4),up_filter_factors=(1, 2, 4),add_output=False)
        self.scascade2 = UNet(num_classes=4,input_channels=1,down_filter_factors=(1, 2, 4),up_filter_factors=(1, 2, 4),add_output=False)
        self.scascade3 = UNet(num_classes=4,input_channels=1,down_filter_factors=(1, 2, 4),up_filter_factors=(1, 2, 4),add_output= True)

    def forward(self,x):
        
        x1 = self.scascade1(x)
        x2 = self.scascade2(x1)
        x3 = self.scascade3(x2)


        return x1,x2,x3
    
########################################### SFTN with Teacher and student ############################################   

class SFTN(nn.Module):

    def __init__(self):

        super(SFTN,self).__init__()

        self.tcascade1 = UNet(num_classes=4,input_channels=1,add_output=False)
        self.tcascade2 = UNet(num_classes=4,input_channels=1,add_output=False)
        self.tcascade3 = UNet(num_classes=4,input_channels=1,add_output= True)
        self.scascade1 = UNet(num_classes=4,input_channels=1,down_filter_factors=(1, 2, 4),up_filter_factors=(1, 2, 4),add_output=False)
        self.scascade2 = UNet(num_classes=4,input_channels=1,down_filter_factors=(1, 2, 4),up_filter_factors=(1, 2, 4),add_output=False)
        self.scascade3 = UNet(num_classes=4,input_channels=1,down_filter_factors=(1, 2, 4),up_filter_factors=(1, 2, 4),add_output= True)
        
    def forward(self,x):
        if self.training: 
            
            x1 = self.tcascade1(x)
            x2 = self.tcascade2(x1)
            x3 = self.tcascade3(x2)

            x1_s = self.scascade2(x1)
            x1_s = self.scascade3(x1_s)

            x2_s = self.scascade3(x2)
#             print(f'x2_s shape is: {x1_s.shape},{x2_s.shape},{x3.shape}')

            op = x1_s,x2_s,x3
        
        else :#(eval mode)
            
            x1 = self.tcascade1(x)           
            x2 = self.tcascade2(x1)
            x3 = self.tcascade3(x2)
#             print(f'x3 shape is: {x3.shape}')
            op = x1,x2,x3
            
        return op
##############################################################################################################################
# class UNet(nn.Module):
    
#     output_downscaled = 1
#     module = UNetModule

#     def __init__(self,
#                  input_channels = 3,
#                  filters_base: int = 32,
#                  down_filter_factors=(1, 2, 4, 8, 16),
#                  up_filter_factors=(1, 2, 4, 8, 16),
#                  bottom_s=4,
#                  num_classes=1,
#                  padding=1,
#                  add_output=True):
#         super().__init__()
#         self.num_classes = num_classes
#         assert len(down_filter_factors) == len(up_filter_factors)
#         assert down_filter_factors[-1] == up_filter_factors[-1]
#         down_filter_sizes = [filters_base * s for s in down_filter_factors]
#         up_filter_sizes = [filters_base * s for s in up_filter_factors]
#         self.down, self.up = nn.ModuleList(), nn.ModuleList()
#         self.down.append(self.module(input_channels, down_filter_sizes[0], padding))
#         for prev_i, nf in enumerate(down_filter_sizes[1:]):
#             self.down.append(self.module(down_filter_sizes[prev_i], nf, padding))
#         for prev_i, nf in enumerate(up_filter_sizes[1:]):
#             self.up.append(self.module(
#                 down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i], padding))
#         pool = nn.MaxPool2d(2, 2)
#         pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
#         upsample = nn.Upsample(scale_factor=2)
#         upsample_bottom = nn.Upsample(scale_factor=bottom_s)
#         self.downsamplers = [None] + [pool] * (len(self.down) - 1)
#         self.downsamplers[-1] = pool_bottom
#         self.upsamplers = [upsample] * len(self.up)
#         self.upsamplers[-1] = upsample_bottom
#         self.add_output = add_output
#         if add_output:
#             self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, padding)

#     def forward(self, x):
#         x1 = self.down[0](x)
#         x2 = self.downsamplers[1](x1)
#         x2 = self.down[1](x2)
#         x3 = self.downsamplers[2](x2)
#         x3 = self.down[2](x3)
#         x4 = self.downsamplers[3](x3)
#         x4 = self.down[3](x4)
#         x5 = self.downsamplers[4](x4)
#         x5 = self.down[4](x5)
#         x6 = self.upsamplers[-1](x5)
#         x6 = self.up[-1](torch.cat([x6, x4], 1))
#         x7 = self.upsamplers[-2](x6)
#         x7 = self.up[-2](torch.cat([x7, x3], 1))        
#         x8 = self.upsamplers[1](x7)
#         x8 = self.up[1](torch.cat([x8, x2], 1))        
#         x9 = self.upsamplers[0](x8)
#         x9 = self.up[0](torch.cat([x9, x1], 1)) 
#         x_out = self.conv_final(x9)
#         x_out = F.log_softmax(x_out, dim=1)
                
#         return x_out
    

class TUNet(nn.Module):
    
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels = 3,
                 filters_base: int = 32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=(1, 2, 4, 8, 16),
                 bottom_s=4,
                 num_classes=1,
                 padding=1,
                 add_output=True):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        
        self.Tdown, self.Tup = nn.ModuleList(), nn.ModuleList()
        self.Tdown.append(self.module(input_channels, down_filter_sizes[0], padding))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.Tdown.append(self.module(down_filter_sizes[prev_i], nf, padding))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.Tup.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i], padding))
        
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        
        self.Tdownsamplers = [None] + [pool] * (len(self.Tdown) - 1)
        self.Tdownsamplers[-1] = pool_bottom
        self.Tupsamplers = [upsample] * len(self.Tup)
        self.Tupsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.Tconv_final = nn.Conv2d(up_filter_sizes[0], num_classes, padding)

    def forward(self, x):
        x1 = self.Tdown[0](x)
        x2 = self.Tdownsamplers[1](x1)
        x2 = self.Tdown[1](x2)
        x3 = self.Tdownsamplers[2](x2)
        x3 = self.Tdown[2](x3)
        x4 = self.Tdownsamplers[3](x3)
        x4 = self.Tdown[3](x4)
        x5 = self.Tdownsamplers[4](x4)
        x5 = self.Tdown[4](x5)
        x6 = self.Tupsamplers[3](x5)
        x6 = self.Tup[3](torch.cat([x6, x4], 1))
        x7 = self.Tupsamplers[2](x6)
        x7 = self.Tup[2](torch.cat([x7, x3], 1))        
        x8 = self.Tupsamplers[1](x7)
        x8 = self.Tup[1](torch.cat([x8, x2], 1))        
        x9 = self.Tupsamplers[0](x8)
        x9 = self.Tup[0](torch.cat([x9, x1], 1)) 
        x_out = self.Tconv_final(x9)
        x_out = F.log_softmax(x_out, dim=1)
                
        return x1,x2,x3,x4,x5,x6,x7,x8,x9,x_out

class SUNet(nn.Module):
    
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels = 3,
                 filters_base: int = 32,
                 down_filter_factors=(1, 2, 4),
                 up_filter_factors=(1, 2, 4),
                 bottom_s=4,
                 num_classes=1,
                 padding=1,
                 add_output=True):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        
        self.Sdown, self.Sup = nn.ModuleList(), nn.ModuleList()
        self.Sdown.append(self.module(input_channels, down_filter_sizes[0], padding))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.Sdown.append(self.module(down_filter_sizes[prev_i], nf, padding))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.Sup.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i], padding))
        
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        
        self.Sdownsamplers = [None] + [pool] * (len(self.Sdown) - 1)
        self.Sdownsamplers[-1] = pool_bottom
        self.Supsamplers = [upsample] * len(self.Sup)
        self.Supsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.Sconv_final = nn.Conv2d(up_filter_sizes[0], num_classes, padding)
    
    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))
            
        if self.add_output:
            x_out = self.conv_final_2(x_out)
            if self.num_classes > 1:
                x_out = F.log_softmax(x_out, dim=1)
        else:
            x_out = self.conv_final_1(x_out)
                
        return x_out

#     def forward(self, x):
#         x1 = self.Sdown[0](x)
#         x2 = self.Sdownsamplers[1](x1)
#         x2 = self.Sdown[1](x2)
#         x3 = self.Sdownsamplers[2](x2)
#         x3 = self.Sdown[2](x3)
        
#         x4 = self.Supsamplers[1](x3)
#         x4 = self.Sup[1](torch.cat([x2, x4], 1))
#         x5 = self.Supsamplers[0](x4)
#         x5 = self.Sup[0](torch.cat([x1, x5], 1))
        
#         x_out = self.Sconv_final(x5)
#         x_out = F.log_softmax(x_out, dim=1)
                
#         return x1,x2,x3,x4,x5,x_out

# class SFTN(nn.Module):
    
#     output_downscaled = 1
#     module = UNetModule

#     def __init__(self,
#                  input_channels = 3,
#                  filters_base: int = 32,
#                  Tdown_filter_factors=(1, 2, 4, 8, 16),
#                  Tup_filter_factors=(1, 2, 4, 8, 16),
#                  Sdown_filter_factors=(1, 2, 4),
#                  Sup_filter_factors=(1, 2, 4),
#                  bottom_s=4,
#                  num_classes=1,
#                  padding=1,
#                  add_output=True):
#         super().__init__()
#         self.num_classes = num_classes
#         assert len(Tdown_filter_factors) == len(Tup_filter_factors)
#         assert Tdown_filter_factors[-1] == Tup_filter_factors[-1]
        
#         Tdown_filter_sizes = [filters_base * s for s in Tdown_filter_factors]
#         Tup_filter_sizes = [filters_base * s for s in Tup_filter_factors]

#         self.Tdown, self.Tup = nn.ModuleList(), nn.ModuleList()
#         self.Tdown.append(self.module(input_channels, Tdown_filter_sizes[0], padding))
#         for prev_i, nf in enumerate(Tdown_filter_sizes[1:]):
#             self.Tdown.append(self.module(Tdown_filter_sizes[prev_i], nf, padding))
#         for prev_i, nf in enumerate(Tup_filter_sizes[1:]):
#             self.Tup.append(self.module(
#                 Tdown_filter_sizes[prev_i] + nf, Tup_filter_sizes[prev_i], padding))

#         pool = nn.MaxPool2d(2, 2)
#         pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
#         upsample = nn.Upsample(scale_factor=2)
#         upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        
#         self.Tdownsamplers = [None] + [pool] * (len(self.Tdown) - 1)
#         self.Tdownsamplers[-1] = pool_bottom
#         self.Tupsamplers = [upsample] * len(self.Tup)
#         self.Tupsamplers[-1] = upsample_bottom
        
#         self.add_output = add_output
#         if add_output:
#             self.Tconv_final = nn.Conv2d(Tup_filter_sizes[0], num_classes, padding)   
            
#         Sdown_filter_sizes = [filters_base * s for s in Sdown_filter_factors]
#         Sup_filter_sizes = [filters_base * s for s in Sup_filter_factors]
        
#         self.Sdown, self.Sup = nn.ModuleList(), nn.ModuleList()
#         self.Sdown.append(self.module(input_channels, Sdown_filter_sizes[0], padding))
#         for prev_i, nf in enumerate(Sdown_filter_sizes[1:]):
#             self.Sdown.append(self.module(Sdown_filter_sizes[prev_i], nf, padding))
#         for prev_i, nf in enumerate(Sup_filter_sizes[1:]):
#             self.Sup.append(self.module(
#                 Sdown_filter_sizes[prev_i] + nf, Sup_filter_sizes[prev_i], padding))        
        
#         self.Sdownsamplers = [None] + [pool] * (len(self.Sdown) - 1)
#         self.Sdownsamplers[-1] = pool_bottom
#         self.Supsamplers = [upsample] * len(self.Sup)
#         self.Supsamplers[-1] = upsample_bottom
        
#         if add_output:
#             self.Sconv_final = nn.Conv2d(Sup_filter_sizes[0], num_classes, padding)
            
#         self.transform1 = nn.Conv2d(64,32,kernel_size=1)
#         self.transform2 = nn.Conv2d(256,64,kernel_size=1)
#         self.transform3 = nn.Conv2d(512,128,kernel_size=1)
#         self.transform4 = nn.Conv2d(128,64,kernel_size=1)

            
#     def forward(self, x):
#         if self.training:
            
#             x1 = self.Tdown[0](x)
#             x2 = self.Tdownsamplers[1](x1)
#             x2 = self.Tdown[1](x2)
#             x3 = self.Tdownsamplers[2](x2)
#             x3 = self.Tdown[2](x3)
#             x4 = self.Tdownsamplers[3](x3)
#             x4 = self.Tdown[3](x4)
#             x5 = self.Tdownsamplers[4](x4)
#             x5 = self.Tdown[4](x5)
#             x6 = self.Tupsamplers[3](x5)
#             x6 = self.Tup[3](torch.cat([x6, x4], 1))
#             x7 = self.Tupsamplers[2](x6)
#             x7 = self.Tup[2](torch.cat([x7, x3], 1))        
#             x8 = self.Tupsamplers[1](x7)
#             x8 = self.Tup[1](torch.cat([x8, x2], 1))        
#             x9 = self.Tupsamplers[0](x8)
#             x9 = self.Tup[0](torch.cat([x9, x1], 1)) 
#             x_out = self.Tconv_final(x9)
#             x_out = F.log_softmax(x_out, dim=1)
            
# #             print(f'teacher shape is {x1.shape},{x2.shape},{x3.shape},{x4.shape},{x5.shape},{x6.shape},{x7.shape},{x8.shape},{x9.shape},{x_out.shape}')

# #             x2_s = F.adaptive_avg_pool2d(x2, (16,32,80,80))
# #             x2_s = nn.Conv2d(in_channels=num_kernels, out_channels=3, kernel_size=1)
#             x2_s = self.transform1(x2)
#             x4_s = self.transform2(x4)
#             x5_s = self.transform3(x5)
#             x5_s = F.adaptive_avg_pool2d(x5_s, (80,80))
#             x7_s = self.transform4(x7)
#             x7_s = F.adaptive_avg_pool2d(x7_s, (160,160))
    
            
#             print(f'shapes after transform are {x2.shape},{x2_s.shape},{x4.shape},{x4_s.shape},{x5.shape},{x5_s.shape},{x7.shape},{x7_s.shape}')
# #             x4_s = F.adaptive_avg_pool2d(x4, (64,64)) 
# #             x5_s = F.adaptive_avg_pool2d(x2, (128,128)) 
# #             x7_s = F.adaptive_avg_pool2d(x2, (64,64)) 
            


# #             s2 = self.Sdownsamplers[1](x2_s)
#             s2 = self.Sdown[1](x2_s)
#             s3 = self.Sdownsamplers[2](s2)
#             s3 = self.Sdown[2](s3)
#             s4 = self.Supsamplers[-1](s3)
#             s4 = self.Sup[-1](torch.cat([s2,s4], 1))
#             s5_1 = self.Supsamplers[-2](s4)
#             s5_1 = self.Sup[-2](torch.cat([x2_s, s5_1], 1))

#             s3 = self.Sdownsamplers[2](x4_s)
#             s3 = self.Sdown[2](s3)
#             s4 = self.Supsamplers[-1](s3)
#             s4 = self.Sup[-1](torch.cat([x4_s,s4], 1))
#             s5_2 = self.Supsamplers[-2](s4)
#             s5_2 = self.Sup[-2](torch.cat([x2_s, s5_2], 1))

#             s4 = self.Supsamplers[-1](x5_s)
#             s4 = self.Sup[-1](torch.cat([x4_s,s4], 1))
#             s5_3 = self.Supsamplers[-2](s4)
#             s5_3 = self.Sup[-2](torch.cat([x2_s, s5_3], 1))

#             s5_4 = self.Supsamplers[-2](x7_s)
#             s5_4 = self.Sup[-2](torch.cat([x2_s, s5_4], 1))
            
#             op = s5_1, s5_2, s5_3, s5_4, x_out
            
#         else:#(eval mode)
#             x1 = self.Tdown[0](x)
#             x2 = self.Tdownsamplers[1](x1)
#             x2 = self.Tdown[1](x2)
#             x3 = self.Tdownsamplers[2](x2)
#             x3 = self.Tdown[2](x3)
#             x4 = self.Tdownsamplers[3](x3)
#             x4 = self.Tdown[3](x4)
#             x5 = self.Tdownsamplers[4](x4)
#             x5 = self.Tdown[4](x5)
#             x6 = self.Tupsamplers[3](x5)
#             x6 = self.Tup[3](torch.cat([x6, x4], 1))
#             x7 = self.Tupsamplers[2](x6)
#             x7 = self.Tup[2](torch.cat([x7, x3], 1))        
#             x8 = self.Tupsamplers[1](x7)
#             x8 = self.Tup[1](torch.cat([x8, x2], 1))        
#             x9 = self.Tupsamplers[0](x8)
#             x9 = self.Tup[0](torch.cat([x9, x1], 1)) 
#             x_out = self.Tconv_final(x9)
#             x_out = F.log_softmax(x_out, dim=1)

#             op = x1,x2,x3,x4,x5,x6,x7,x8,x9,x_out
        
#         return op
# #         return x_out

    
    
# class SUNet2(nn.Module):
    
#     output_downscaled = 1
#     module = UNetModule

#     def __init__(self,
#                  input_channels = 3,
#                  filters_base: int = 32,
#                  down_filter_factors=(1, 2, 4, 8, 16),
#                  up_filter_factors=(1, 2, 4, 8, 16),
#                  bottom_s=4,
#                  num_classes=1,
#                  padding=1,
#                  add_output=True):
#         super().__init__()
#         self.num_classes = num_classes
#         assert len(down_filter_factors) == len(up_filter_factors)
#         assert down_filter_factors[-1] == up_filter_factors[-1]
#         down_filter_sizes = [filters_base * s for s in down_filter_factors]
#         up_filter_sizes = [filters_base * s for s in up_filter_factors]
#         self.down, self.up = nn.ModuleList(), nn.ModuleList()
#         self.down.append(self.module(input_channels, down_filter_sizes[0], padding))
#         for prev_i, nf in enumerate(down_filter_sizes[1:]):
#             self.down.append(self.module(down_filter_sizes[prev_i], nf, padding))
#         for prev_i, nf in enumerate(up_filter_sizes[1:]):
#             self.up.append(self.module(
#                 down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i], padding))
#         pool = nn.MaxPool2d(2, 2)
#         pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
#         upsample = nn.Upsample(scale_factor=2)
#         upsample_bottom = nn.Upsample(scale_factor=bottom_s)
# #         self.downsamplers = [None] + [pool] * (len(self.down) - 1)
#         self.downsamplers = [None] + [pool] + [None] +[pool] +[pool]
#         self.downsamplers[-1] = pool_bottom
# #         self.upsamplers = [upsample] * len(self.up)
#         self.upsamplers = [upsample] + [None] +[upsample] +[upsample]
#         self.upsamplers[-1] = upsample_bottom
#         self.add_output = add_output
#         if add_output:
#             self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, padding)

#     def forward(self, x):
#         xs = []
#         x_out = self.down[0](x)
#         xs.append(x_out)
        
#         for downsample, down in zip(self.downsamplers[1:], self.down[1:]):
#             x_in = xs[-1] if downsample is None else downsample(xs[-1])
#             x_out = down(x_in)
#             xs.append(x_out)

#         x_out = xs[-1]
#         for x_skip, upsample, up in reversed(
#                 list(zip(xs[:-1], self.upsamplers, self.up))):
#             x_out = x_out if upsample is None else upsample(x_out)
# #             x_out = upsample(x_out)
#             x_out = up(torch.cat([x_out, x_skip], 1))

#         if self.add_output:
#             x_out = self.conv_final(x_out)
#             if self.num_classes > 1:
#                 x_out = F.log_softmax(x_out, dim=1)
                
#         return x_out

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

####################################################### INNER BLOCKS #####################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
#################################################################### BASIC_UNET ##########################################################
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, add_output=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.add_output = add_output

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.conv2 = nn.Conv2d(64,1, kernel_size=1)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.add_output:
            logits = self.outc(x)
            x_out = F.log_softmax(logits, dim=1)
        else:
            x_out = self.conv2(x)
        return x_out
#         return logits
    
######################################################## Teacher UNet D5C5 #########################################################
class DCTeacherUNet(nn.Module):

    def __init__(self):

        super(DCTeacherUNet,self).__init__()

        self.tcascade1 = UNet(n_channels=1, n_classes=4,add_output=False)
        self.tcascade2 = UNet(n_channels=1, n_classes=4,add_output=False)
        self.tcascade3 = UNet(n_channels=1, n_classes=4,add_output= True)

    def forward(self,x):
        
        x1 = self.tcascade1(x)
        x2 = self.tcascade2(x1)
        x3 = self.tcascade3(x2)


        return x1,x2,x3

#########################################################  Unet_depth -3 ###############################################################


class UNet_S(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, add_output=True):
        super(UNet_S, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.add_output = add_output

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down2 = Down(128, 256//factor)
        self.up1 = Up(256,128//factor,bilinear)
        self.up2 = Up(128,64,bilinear)
        self.outc = OutConv(64,n_classes)
        self.conv2 = nn.Conv2d(64,1,kernel_size=1)       

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)   

        if self.add_output:
            logits = self.outc(x)
            x_out = F.log_softmax(logits, dim=1)
        else:
            x_out = self.conv2(x)
        return x_out

########################################### CASCADED Student UNET with depth 3 #######################################################
   
class DCStudentUNet(nn.Module):

    def __init__(self):

        super(DCStudentUNet,self).__init__()

        self.scascade1 = UNet_S(n_channels=1, n_classes=4, add_output=False)
        self.scascade2 = UNet_S(n_channels=1, n_classes=4, add_output=False)
        self.scascade3 = UNet_S(n_channels=1, n_classes=4, add_output=True)

    def forward(self,x):
        
        x1 = self.scascade1(x)
        x2 = self.scascade2(x1)
        x3 = self.scascade3(x2)


        return x1,x2,x3   
    
########################################### SFTN with Teacher and student ###########################################################   

class SFTN(nn.Module):

    def __init__(self):

        super(SFTN,self).__init__()

        self.tcascade1 = UNet(n_channels=1, n_classes=4,add_output=False)
        self.tcascade2 = UNet(n_channels=1, n_classes=4,add_output=False)
        self.tcascade3 = UNet(n_channels=1, n_classes=4,add_output= True)
        self.scascade1 = UNet_S(n_channels=1, n_classes=4, add_output=False)
        self.scascade2 = UNet_S(n_channels=1, n_classes=4, add_output=False)
        self.scascade3 = UNet_S(n_channels=1, n_classes=4, add_output=True)
        
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

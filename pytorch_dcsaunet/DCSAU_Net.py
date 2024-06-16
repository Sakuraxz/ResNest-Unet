import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_dcsaunet.encoder import CSA

csa_block = CSA()

class Up(nn.Module):
    """Upscaling"""
    def __init__(self):
        super().__init__()       
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

    
class PFC(nn.Module):
    def __init__(self,channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(3, channels, kernel_size, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x
    
# 引入注意力机制，动态地学习每个通道的重要性
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        out = self.squeeze(x).view(batch_size, channels)
        out = self.excitation(out).view(batch_size, channels, 1, 1)
        
        return x * out.expand_as(x)
    
#Attention and Concatenation Block    
class ANCBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ANCBlock, self).__init__()
        self.se_block = SEBlock(in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels // 2,kernel_size=1, stride=1, padding=0)
        
    def forward(self, x1, x2):
        x3 = self.conv(self.se_block(x1))
        x4 = self.conv(self.se_block(x2))
        out = torch.cat([x3, x4], dim=1)
        return out
        

# inherit nn.module
class Model(nn.Module):
    def __init__(self,img_channels=3, n_classes=1):
        super(Model, self).__init__()
        self.img_channels = img_channels
        self.n_classes = n_classes
        self.pfc = PFC(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.up_conv1 = Up()
        self.up_conv2 = Up()
        self.up_conv3 = Up()
        self.up_conv4 = Up()
        self.down1 = csa_block.layer1
        self.down2 = csa_block.layer2
        self.down3 = csa_block.layer3
        self.down4 = csa_block.layer4
        self.up1 = csa_block.layer5
        self.up2 = csa_block.layer6
        self.up3 = csa_block.layer7
        self.up4 = csa_block.layer8
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
        
        self.anc_block1 = ANCBlock(512)
        self.anc_block2 = ANCBlock(256)
        self.anc_block3 = ANCBlock(128)
        self.anc_block4 = ANCBlock(64)
        
        self.conv1 = nn.Conv2d(512,256,kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256,128,kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 64,kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 32,kernel_size=1, stride=1, padding=0)
        
       
    
    def forward(self, x): #最初的版本
        x1 = self.pfc(x)
        x2 = self.maxpool(x1)
        x3 = self.down1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)
        x9 = self.down4(x8)
        x10 = self.up_conv1(x9, x7)
        x11 = self.up1(x10)
        x12 = self.up_conv2(x11, x5)
        x13 = self.up2(x12)
        x14 = self.up_conv3(x13, x3)
        x15 = self.up3(x14)
        x16 = self.up_conv4(x15, x1)
        x17 = self.up4(x16)
        x18 = self.out_conv(x17) 
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(x4.shape)
        print(x5.shape)
        print(x6.shape)
        print(x7.shape)
        print(x8.shape)
        print(x9.shape)
        print(x10.shape)
        print(x11.shape)
        print(x12.shape)
        print(x13.shape)
        print(x14.shape)
        print(x15.shape)
        print(x16.shape)
        print(x17.shape)
        print(x18.shape)
        input("---")
        return x18
#     def forward(self, x): #修改跳跃连接

#         x1 = self.pfc(x)
#         x2 = self.maxpool(x1)
#         x3 = self.down1(x2)
#         x4 = self.maxpool(x3)
#         x5 = self.down2(x4)
#         x6 = self.maxpool(x5)
#         x7 = self.down3(x6)
#         x8 = self.maxpool(x7)
#         x9 = self.down4(x8)
#         x19 = self.conv1(x9) 
#         x20 = self.conv1(x8)
#         x21 = torch.cat([x19, x20], dim=1)
#         x10 = self.up_conv1(x21, x7)
#         x11 = self.up1(x10)
#         x22 = self.conv2(x11)
#         x23 = self.conv2(x6)
#         x24 = torch.cat([x22, x23], dim=1)
#         x12 = self.up_conv2(x24, x5)
#         x13 = self.up2(x12)
#         x25 = self.conv3(x13)
#         x26 = self.conv3(x4)
#         x27 = torch.cat([x25, x26], dim=1)
#         x14 = self.up_conv3(x27, x3)
#         x15 = self.up3(x14)
#         x28 = self.conv4(x15)
#         x29 = self.conv4(x2)
#         x30 = torch.cat([x28, x29], dim=1)
#         x16 = self.up_conv4(x30, x1)
#         x17 = self.up4(x16)
#         x18 = self.out_conv(x17)

#         return x18
#     def forward(self, x):

#         x1 = self.pfc(x)
#         x2 = self.maxpool(x1)
#         x3 = self.down1(x2)
#         x4 = self.maxpool(x3)
#         x5 = self.down2(x4)
#         x6 = self.maxpool(x5)
#         x7 = self.down3(x6)
#         x8 = self.maxpool(x7)
#         x9 = self.down4(x8)
#         x10 = self.anc_block1(x9,x8)
#         x11 = self.up_conv1(x10, x7)
#         x12 = self.up1(x11)
#         x13 = self.anc_block2(x12,x6)
#         x14 = self.up_conv2(x13, x5)
#         x15 = self.up2(x14)
#         x16 = self.anc_block3(x15,x4)
#         x17 = self.up_conv3(x16, x3)
#         x18 = self.up3(x17)
#         x19 = self.anc_block4(x18,x2)
#         x20 = self.up_conv4(x19, x1)
#         x21 = self.up4(x20)
#         x22 = self.out_conv(x21)
        
#         return x22
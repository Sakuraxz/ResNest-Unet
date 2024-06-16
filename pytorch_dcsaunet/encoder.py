"""
Reference:

- Zhang, Hang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun et al. "Resnest: Split-attention networks." arXiv preprint arXiv:2004.08955 (2020)
"""
 
import torch
from pytorch_dcsaunet import resnet

 
ResNet = resnet.ResNet
Bottleneck = resnet.Bottleneck

def CSA(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
   
    return model
 
# Bottleneck: 残差块类型，这里选择的是Bottleneck结构，通常用于深层网络中，包括了两个3x3的卷积层以及一个恒等映射或者投影映射。

# [2, 2, 2, 2]: 每个stage中重复的残差块数量，这里表示在每个stage中有2个残差块。

# radix=2: Radix是指分组卷积（group convolution）中每个分组的通道数。

# groups=1: 分组卷积中的组数。

# bottleneck_width=64: Bottleneck内部通道数，即Bottleneck结构中第二个3x3卷积层的输出通道数。

# deep_stem=True: 是否使用深度卷积作为网络的输入处理模块。

# stem_width=32: 输入处理模块中的输出通道数。

# avg_down=True: 是否使用平均池化作为下采样操作。

# avd=True: 是否使用自适应卷积(downsample with stride).

# avd_first=False: 自适应卷积(downsample with stride)是否放在基础块的第一层。

# **kwargs: 其他可选参数。

 
 
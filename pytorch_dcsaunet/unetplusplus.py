import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class model(nn.Module):
    def __init__(self, img_channels, n_classes):
        super(model, self).__init__()
        self.encoder = nn.ModuleList([
            ConvBlock(img_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        ])

        self.decoder = nn.ModuleList([
            ConvBlock(512 + 256, 256),
            ConvBlock(256 + 128, 128),
            ConvBlock(128 + 64, 64)
        ])

        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        for block in self.encoder:
            x = block(x)
            encoder_outputs.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        for i, block in enumerate(self.decoder):
            x = self.upconv[i](x)
            x = torch.cat([x, encoder_outputs[-(i + 2)]], dim=1)
            x = block(x)

        x = self.final_conv(x)
        return x
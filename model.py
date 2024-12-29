import torch
import torch.nn as nn

class Recursive3DCNN(nn.Module):
    def __init__(self):
        super(Recursive3DCNN, self).__init__()

        self.encoder1 = Conv3DBlock(2, 16)
        self.encoder2 = Conv3DBlock(16, 32)
        self.encoder3 = Conv3DBlock(32, 64)
        self.encoder4 = Conv3DBlock(64, 128)
        self.encoder5 = Conv3DBlock(128, 256)

        self.bottleneck = Conv3DBlock(256, 512)

        self.decoder5 = ConvTranspose3DBlock(512, 256)
        self.decoder4 = ConvTranspose3DBlock(256, 128)
        self.decoder3 = ConvTranspose3DBlock(128, 64)
        self.decoder2 = ConvTranspose3DBlock(64, 32)
        self.decoder1 = ConvTranspose3DBlock(32, 16)

        self.output_layer = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        b = self.bottleneck(e5)

        d5 = self.decoder5(b) + e5
        d4 = self.decoder4(d5) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1

        output = torch.sigmoid(self.output_layer(d1))
        output = output.squeeze(2)
        return output


class Conv3DBlock(nn.Module):
    """3D Convolution Block with InstanceNorm and GeLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class ConvTranspose3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(ConvTranspose3DBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.norm(self.conv_transpose(x)))


import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(DoubleConv, self).__init__()
        # two conv layers with relu
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # apply conv->relu twice
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class MaxPooling(nn.Module):
    def __init__(self, kernel_size = 2, stride= 2):
        super(MaxPooling, self).__init__()
        # downsampling by 2
        self.maxPooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxPooling(x)

class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, stride= 2):
        super(UpConv2d, self).__init__()
        # upsampling by 2
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        return self.upconv(x)

def crop_tensor(source, target):
    # crop source to match target size
    source_size = source.size()[2:]
    target_size = target.size()[2:]
    target_h, target_w = target_size
    source_h, source_w = source_size
    
    # calculate center crop
    delta_h = source_h - target_h
    delta_w = source_w - target_w
    start_h = delta_h // 2
    start_w = delta_w // 2

    # perform cropping
    encoder_crop = source[:, :, start_h:start_h + target_h, start_w:start_w + target_w]
    
    return encoder_crop


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # encoder (contracting path)
        self.convBlock1 = DoubleConv(in_channels=1, out_channels=64, kernel_size=3)
        self.convBlock2 = DoubleConv(in_channels=64, out_channels=128, kernel_size=3)
        self.convBlock3 = DoubleConv(in_channels=128, out_channels=256, kernel_size=3)
        self.convBlock4 = DoubleConv(in_channels=256, out_channels=512, kernel_size=3)
        self.convBlock5 = DoubleConv(in_channels=512, out_channels=1024, kernel_size=3)
        self.maxPoolingBlock = MaxPooling(kernel_size=2, stride=2)

        # decoder (expansive path)
        self.upconv2d1 = UpConv2d(in_channels=1024, out_channels=512, kernel_size = 2, stride= 2)
        self.convBlock1_decoder =  DoubleConv(in_channels=1024, out_channels=512, kernel_size=3)

        self.upconv2d2 = UpConv2d(in_channels=512, out_channels=256, kernel_size = 2, stride= 2)
        self.convBlock2_decoder =  DoubleConv(in_channels=512, out_channels=256, kernel_size=3)

        self.upconv2d3 = UpConv2d(in_channels=256, out_channels=128, kernel_size = 2, stride= 2)
        self.convBlock3_decoder =  DoubleConv(in_channels=256, out_channels=128, kernel_size=3)

        self.upconv2d4 = UpConv2d(in_channels=128, out_channels=64, kernel_size = 2, stride= 2)
        self.convBlock4_decoder =  DoubleConv(in_channels=128, out_channels=64, kernel_size=3)

        # final classification layer
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # encoder (contracting path)
        x1 = self.convBlock1(x)        # first conv block
        x2 = self.maxPoolingBlock(x1)  # downsample
        x3 = self.convBlock2(x2)
        x4 = self.maxPoolingBlock(x3)
        x5 = self.convBlock3(x4)
        x6 = self.maxPoolingBlock(x5)
        x7 = self.convBlock4(x6)
        x8 = self.maxPoolingBlock(x7)
        x9 = self.convBlock5(x8)       # bottleneck

        # skip_connections ==> [x1, x3, x5, x7] - bottlenec ==> x9

        # decoder (expansive path)
        x10 = self.upconv2d1(x9)                          # upsample
        cropx7 = crop_tensor(x7, x10)                     # crop skip connection
        x11 = self.convBlock1_decoder(torch.cat((x10, cropx7), 1))  # concatenate and conv

        x12 = self.upconv2d2(x11)
        cropx5 = crop_tensor(x5, x12)
        x13 = self.convBlock2_decoder(torch.cat((x12, cropx5), 1))

        x14 = self.upconv2d3(x13)
        cropx3 = crop_tensor(x3, x14)
        x15 = self.convBlock3_decoder(torch.cat((x14, cropx3), 1))

        x16 = self.upconv2d4(x15)
        cropx1 = crop_tensor(x1, x16)
        x17 = self.convBlock4_decoder(torch.cat((x16, cropx1), 1))

        x18 = self.conv1x1(x17)        # final output

        return x18
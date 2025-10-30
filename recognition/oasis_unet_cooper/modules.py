import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.do = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, dropout=dropout),
            ConvBNAct(out_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad to match x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    """
    A clean U-Net for 2D segmentation with dropout & BN.
    """
    def __init__(self, in_channels=1, num_classes=4, base_ch=32, bilinear=True, dropout=0.1):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_ch, dropout=dropout)
        self.down1 = Down(base_ch, base_ch*2, dropout=dropout)
        self.down2 = Down(base_ch*2, base_ch*4, dropout=dropout)
        self.down3 = Down(base_ch*4, base_ch*8, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_ch*8, base_ch*16 // factor, dropout=dropout)

        self.up1 = Up(base_ch*16, base_ch*8 // factor, bilinear=bilinear, dropout=dropout)
        self.up2 = Up(base_ch*8, base_ch*4 // factor, bilinear=bilinear, dropout=dropout)
        self.up3 = Up(base_ch*4, base_ch*2 // factor, bilinear=bilinear, dropout=dropout)
        self.up4 = Up(base_ch*2, base_ch, bilinear=bilinear, dropout=dropout)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits
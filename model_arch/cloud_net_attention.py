import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.gating = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        g = self.gating(g)
        combined = torch.cat([x, g], dim=1)
        attention = self.attention(combined)
        return x * attention

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(ResidualConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        return nn.ReLU()(out + residual)

class CloudNetImproved(nn.Module):
    def __init__(self, input_rows=512, input_cols=512, num_of_channels=4, num_of_classes=1):
        super(CloudNetImproved, self).__init__()
        # Encoder
        self.conv1 = ResidualConvBlock(num_of_channels, 32)  # [B, 32, 512, 512]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ResidualConvBlock(32, 64)  # [B, 64, 256, 256]
        self.conv3 = ResidualConvBlock(64, 128)  # [B, 128, 128, 128]
        self.conv4 = ResidualConvBlock(128, 256)  # [B, 256, 64, 64]

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )

        # Decoder
        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att4 = AttentionGate(512, 256)  # Attention on conv4 (256) for bottleneck (512)
        self.dec4 = ResidualConvBlock(512 + 256, 256)  # 512 (up4) + 256 (att4) = 768 -> 256
        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionGate(256, 128)  # Attention on conv3 (128) for dec4 (256)
        self.dec3 = ResidualConvBlock(256 + 128, 128)  # 256 (up3) + 128 (att3) = 384 -> 128
        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(128, 64)  # Attention on conv2 (64) for dec3 (128)
        self.dec2 = ResidualConvBlock(128 + 64, 64)  # 128 (up2) + 64 (att2) = 192 -> 64
        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(64, 32)  # Attention on conv1 (32) for dec2 (64)
        self.dec1 = ResidualConvBlock(64 + 32, 32)  # 64 (up1) + 32 (att1) = 96 -> 32

        self.final = nn.Conv2d(32, num_of_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)  # [B, 32, 512, 512]
        pool1 = self.pool(conv1)  # [B, 32, 256, 256]
        conv2 = self.conv2(pool1)  # [B, 64, 256, 256]
        pool2 = self.pool(conv2)  # [B, 64, 128, 128]
        conv3 = self.conv3(pool2)  # [B, 128, 128, 128]
        pool3 = self.pool(conv3)  # [B, 128, 64, 64]
        conv4 = self.conv4(pool3)  # [B, 256, 64, 64]
        pool4 = self.pool(conv4)  # [B, 256, 32, 32]

        # Bottleneck
        bottleneck = self.bottleneck(pool4)  # [B, 512, 32, 32]

        # Decoder
        up4 = self.upconv4(bottleneck)  # [B, 512, 64, 64]
        att4 = self.att4(conv4, up4)  # [B, 256, 64, 64]
        up4 = torch.cat([up4, att4], dim=1)  # [B, 512 + 256 = 768, 64, 64]
        dec4 = self.dec4(up4)  # [B, 256, 64, 64]
        up3 = self.upconv3(dec4)  # [B, 256, 128, 128]
        att3 = self.att3(conv3, up3)  # [B, 128, 128, 128]
        up3 = torch.cat([up3, att3], dim=1)  # [B, 256 + 128 = 384, 128, 128]
        dec3 = self.dec3(up3)  # [B, 128, 128, 128]
        up2 = self.upconv2(dec3)  # [B, 128, 256, 256]
        att2 = self.att2(conv2, up2)  # [B, 64, 256, 256]
        up2 = torch.cat([up2, att2], dim=1)  # [B, 128 + 64 = 192, 256, 256]
        dec2 = self.dec2(up2)  # [B, 64, 256, 256]
        up1 = self.upconv1(dec2)  # [B, 64, 512, 512]
        att1 = self.att1(conv1, up1)  # [B, 32, 512, 512]
        up1 = torch.cat([up1, att1], dim=1)  # [B, 64 + 32 = 96, 512, 512]
        dec1 = self.dec1(up1)  # [B, 32, 512, 512]

        final = self.final(dec1)  # [B, num_of_classes, 512, 512]
        return torch.sigmoid(final)

def model_arch(input_rows=512, input_cols=512, num_of_channels=4, num_of_classes=1):
    return CloudNetImproved(input_rows, input_cols, num_of_channels, num_of_classes)
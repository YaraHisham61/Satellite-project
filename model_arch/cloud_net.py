import torch
import torch.nn as nn
import torch.nn.functional as F

class CloudNet(nn.Module):
    def __init__(self, input_rows=512, input_cols=512, num_of_channels=4, num_of_classes=1):
        super(CloudNet, self).__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_of_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128 (skip) + 128 (upconv)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.final = nn.Conv2d(16, num_of_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)  # [B, 16, 512, 512]
        pool1 = self.pool(conv1)  # [B, 16, 256, 256]
        conv2 = self.conv2(pool1)  # [B, 32, 256, 256]
        pool2 = self.pool(conv2)  # [B, 32, 128, 128]
        conv3 = self.conv3(pool2)  # [B, 64, 128, 128]
        pool3 = self.pool(conv3)  # [B, 64, 64, 64]
        conv4 = self.conv4(pool3)  # [B, 128, 64, 64]
        pool4 = self.pool(conv4)  # [B, 128, 32, 32]
        # Bottleneck
        bottleneck = self.bottleneck(pool4)  # [B, 256, 32, 32]
        # Decoder
        up4 = self.upconv4(bottleneck)  # [B, 128, 64, 64]
        up4 = torch.cat([up4, conv4], dim=1)  # [B, 256, 64, 64]
        dec4 = self.dec4(up4)  # [B, 128, 64, 64]
        up3 = self.upconv3(dec4)  # [B, 64, 128, 128]
        up3 = torch.cat([up3, conv3], dim=1)  # [B, 128, 128, 128]
        dec3 = self.dec3(up3)  # [B, 64, 128, 128]
        up2 = self.upconv2(dec3)  # [B, 32, 256, 256]
        up2 = torch.cat([up2, conv2], dim=1)  # [B, 64, 256, 256]
        dec2 = self.dec2(up2)  # [B, 32, 256, 256]
        up1 = self.upconv1(dec2)  # [B, 16, 512, 512]
        up1 = torch.cat([up1, conv1], dim=1)  # [B, 32, 512, 512]
        dec1 = self.dec1(up1)  # [B, 16, 512, 512]
        final = self.final(dec1)  # [B, 1, 512, 512]
        return torch.sigmoid(final)

def model_arch(input_rows=512, input_cols=512, num_of_channels=4, num_of_classes=1):
    return CloudNet(input_rows, input_cols, num_of_channels, num_of_classes)
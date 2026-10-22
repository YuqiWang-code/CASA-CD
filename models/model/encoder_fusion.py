"""Hierarchical feature extraction and aggregation (HFEA); see models/SOURCES.md."""
import torch
from torch import nn

class EncoderFusion(nn.Module):
    def __init__(self, inc, midc=32, outc=64):
        super(EncoderFusion, self).__init__()

        if inc is None:
            inc = [16, 24, 32, 96, 320]
        self.inc = inc
        self.midc = midc
        self.outc = outc
        self.fusec = [midc * 3, midc * 3, midc * 3, midc * 2]


        # stage 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.inc[0], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.inc[1], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 2
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(self.inc[1], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 3
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(self.inc[4], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 4
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(self.inc[4], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )

        # aggregation
        self.aggregation_s1 = AggregationModule(self.fusec[0], self.inc[1], self.outc)
        self.aggregation_s2 = AggregationModule(self.fusec[1], self.inc[2], self.outc)
        self.aggregation_s3 = AggregationModule(self.fusec[2], self.inc[3], self.outc)
        self.aggregation_s4 = AggregationModule(self.fusec[3], self.inc[4], self.outc)

    def forward(self, x1, x2, x3, x4, x5):
        s1_f1 = self.conv1_1(x1)
        s1_f2 = self.conv1_2(x2)
        s1_f3 = self.conv1_3(x3)
        s1 = self.aggregation_s1(torch.cat([s1_f1, s1_f2, s1_f3], dim=1), x2)

        s2_f1 = self.conv2_1(x2)
        s2_f2 = self.conv2_2(x3)
        s2_f3 = self.conv2_3(x4)
        s2 = self.aggregation_s2(torch.cat([s2_f1, s2_f2, s2_f3], dim=1), x3)

        s3_f1 = self.conv3_1(x3)
        s3_f2 = self.conv3_2(x4)
        s3_f3 = self.conv3_3(x5)
        s3 = self.aggregation_s3(torch.cat([s3_f1, s3_f2, s3_f3], dim=1), x4)

        s4_f1 = self.conv4_1(x4)
        s4_f2 = self.conv4_2(x5)
        s4 = self.aggregation_s4(torch.cat([s4_f1, s4_f2], dim=1), x5)
        return s1, s2, s3, s4


# feature aggregation
class AggregationModule(nn.Module):
    def __init__(self, fusec, inc, outc):
        super(AggregationModule, self).__init__()
        self.fusec = fusec
        self.inc = inc
        self.outc = outc

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fusec, self.outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.outc, self.outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.outc)
        )

        self.conv_identity = nn.Conv2d(self.inc, self.outc, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))
        return c_out

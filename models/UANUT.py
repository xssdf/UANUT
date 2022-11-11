import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_ch):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_ch, 768)
        self.fc2 = nn.Linear(768, in_ch)

        self.gelu = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class CTM(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super(CTM, self).__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, 
                                          kernel_size=int(scale / 16), stride=int(scale / 16))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, in_ch))
        self.atte_norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.atte = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.ffn_norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.ffn = MLP(in_ch)
        self.enco_norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.outconv = BaseConv(in_ch, out_ch)

    def forward(self, x):

        x1 = x

        #  embedding
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2) + self.position_embeddings

        #  1D conv
        hx = x
        x = self.att(self.atte_norm(x)) + hx

        #  MLP
        hx = x
        x = self.ffn(self.ffn_norm(x)) + hx

        #  reshape
        x = self.enco_norm(x)
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        x = self.outconv(x)
        x = _upsample_like(x, x1)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(out_ch, in_ch, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x


class CNNAttetion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CNNAttetion, self).__init__()

        self.channel_att = ChannelAttention(in_ch, out_ch)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


class BaseConv(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(BaseConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv1(x1)
        return self.conv2(x2) + x1


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src


class UAM1(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM1, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv3 = BaseConv(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv4 = BaseConv(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv5 = BaseConv(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv6 = BaseConv(mid_ch, mid_ch)

        self.BaseConv6d = BaseConv(mid_ch, mid_ch)
        self.BaseConv5d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv4d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv3d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv2d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.BaseConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.BaseConv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.BaseConv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.BaseConv6(hx)

        hx6d = self.BaseConv6d(hx6)
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.BaseConv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.BaseConv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.BaseConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.BaseConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM2(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM2, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv3 = BaseConv(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv4 = BaseConv(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv5 = BaseConv(mid_ch, mid_ch)

        self.BaseConv5d = BaseConv(mid_ch, mid_ch)
        self.BaseConv4d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv3d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv2d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.BaseConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.BaseConv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.BaseConv5(hx)

        hx5d = self.BaseConv5d(hx5)
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.BaseConv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.BaseConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.BaseConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM3(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM3, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv3 = BaseConv(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv4 = BaseConv(mid_ch, mid_ch)

        self.BaseConv4d = BaseConv(mid_ch, mid_ch)
        self.BaseConv3d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv2d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.BaseConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.BaseConv4(hx)

        hx4d = self.BaseConv4d(hx4)
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.BaseConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.BaseConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM4, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv3 = BaseConv(mid_ch, mid_ch)

        self.BaseConv3d = BaseConv(mid_ch, mid_ch)
        self.BaseConv2d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.BaseConv3(hx)

        hx3d = self.BaseConv3d(hx3)
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.BaseConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM5, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)

        self.BaseConv2d = BaseConv(mid_ch, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)

        hx2d = self.BaseConv2d(hx2)
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM1D(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM1D, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv3 = BaseConv(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv4 = BaseConv(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv5 = BaseConv(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv6 = BaseConv(mid_ch, mid_ch)

        self.BaseConv6d = BaseConv(mid_ch, mid_ch)
        self.BaseConv5d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv4d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv3d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv2d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.BaseConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.BaseConv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.BaseConv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.BaseConv6(hx)

        hx6d = self.BaseConv6d(hx6)
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.BaseConv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.BaseConv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.BaseConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.BaseConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM2D(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM2D, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv3 = BaseConv(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv4 = BaseConv(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv5 = BaseConv(mid_ch, mid_ch)

        self.BaseConv5d = BaseConv(mid_ch, mid_ch)
        self.BaseConv4d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv3d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv2d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.BaseConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.BaseConv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.BaseConv5(hx)

        hx5d = self.BaseConv5d(hx5)
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.BaseConv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.BaseConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.BaseConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM3D(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM3D, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv3 = BaseConv(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv4 = BaseConv(mid_ch, mid_ch)

        self.BaseConv4d = BaseConv(mid_ch, mid_ch)
        self.BaseConv3d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv2d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.BaseConv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.BaseConv4(hx)

        hx4d = self.BaseConv4d(hx4)
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.BaseConv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.BaseConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM4D(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM4D, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv3 = BaseConv(mid_ch, mid_ch)

        self.BaseConv3d = BaseConv(mid_ch, mid_ch)
        self.BaseConv2d = BaseConv(mid_ch * 2, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.BaseConv3(hx)

        hx3d = self.BaseConv3d(hx3)
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.BaseConv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class UAM5D(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(UAM5D, self).__init__()

        self.BaseConvin = BaseConv(in_ch, out_ch)

        self.BaseConv1 = BaseConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.BaseConv2 = BaseConv(mid_ch, mid_ch)

        self.BaseConv2d = BaseConv(mid_ch, mid_ch)
        self.BaseConv1d = BaseConv(mid_ch * 2, out_ch)

        self.atte = CNNAttetion(out_ch, out_ch)

    def forward(self, x):
        hx = x

        hxin = self.BaseConvin(hx)

        hx1 = self.BaseConv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.BaseConv2(hx)

        hx2d = self.BaseConv2d(hx2)
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.BaseConv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.atte(hx1d)

        return hx1d + hxin


class NUT(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(NUT, self).__init__()

        self.stage1 = UAM1(in_ch, 16, 64)
        self.pool12 = nn.AvgPool2d(2, stride=2, ceil_mode=True)
        self.trans1 = CTM(64, 64, 256)

        self.stage2 = UAM2(64, 32, 128)
        self.pool23 = nn.AvgPool2d(2, stride=2, ceil_mode=True)
        self.trans2 = CTM(128, 128, 128)

        self.stage3 = UAM3(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.trans3 = CTM(256, 256, 64)

        self.stage4 = UAM4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.trans4 = CTM(512, 512, 32)

        self.stage5 = UAM5(512, 256, 512)
        self.trans5 = CTM(512, 512, 16)

        self.stage5d = UAM5D(512, 256, 512)
        self.stage4d = UAM4D(1024, 128, 256)
        self.stage3d = UAM3D(512, 64, 128)
        self.stage2d = UAM2D(256, 32, 64)
        self.stage1d = UAM1D(128, 16, 32)

        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(out_ch * 5, out_ch, kernel_size=1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx12 = self.trans1(hx1)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx22 = self.trans2(hx2)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx32 = self.trans3(hx3)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx42 = self.trans4(hx4)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx52 = self.trans5(hx)

        # -------------------- decoder --------------------

        hx5d = self.stage5d(hx52)
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx42), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx32), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx22), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx12), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5)
        d5 = _upsample_like(d5, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))

        return d0

        # if deep_s == True:
        #     return d0, d1, d2, d3, d4, d5

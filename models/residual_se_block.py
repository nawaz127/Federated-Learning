import torch
import torch.nn as nn

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        nn.utils.weight_norm(self.conv2)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.dropout2 = nn.Dropout2d(0.2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.drop_path = nn.Identity() if drop_path_rate == 0 else DropPath(drop_path_rate)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            nn.utils.weight_norm(shortcut_conv)
            self.shortcut = nn.Sequential(
                shortcut_conv,
                nn.GroupNorm(8, out_channels)
            )
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        se_weight = self.se(out)
        out = out * se_weight
        out = self.drop_path(out) + shortcut
        out = self.relu(out)
        return out

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

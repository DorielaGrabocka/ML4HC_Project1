from typing import List

import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, residual: bool = False) -> None:
        super().__init__()
        self.residual = residual

        padding = "same"
        if stride == 2:
            padding = 0

        # handles downsampling/filter matching if necessary
        self.adapter = None
        if (in_channels != out_channels or stride == 2) and (self.residual is True):
            self.adapter = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.adapter is not None:
            identity = self.adapter(x)

        if self.residual:
            out += identity
        out = self.relu(out)

        return out


class CNN(nn.Module):

    def __init__(self, n_classes: int, n_filters: List[int], kernel_sizes: List[int], strides: List[int],
                 fully_connected_features: int, adaptive_average_len: int, residual: bool) -> None:
        """
        n_filters: convolutional filters number for each conv block, also gives the number of conv blocks
        kernel_sizes: the convolutional filter size of each conv block
        strides: the stride of each conv block, max 2 blocks should be strided because the inp lenght is quite short (188)
        fully_connected_features: number of hidden units in the fully connected layer
        residual: flag specifying whether residual connections should be used
        """
        super().__init__()
        assert len(kernel_sizes) == len(strides) and len(strides) == len(n_filters)
        self.n_classes = n_classes

        conv_blocks = []
        in_channels = 1
        for filters, kernel_size, stride in zip(n_filters, kernel_sizes, strides):
            conv_blocks.append(ConvBlock(in_channels, filters, kernel_size, stride, residual))
            in_channels = filters

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.pool = nn.AdaptiveAvgPool1d(adaptive_average_len)
        self.linear1 = nn.Linear(filters * adaptive_average_len, fully_connected_features)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(fully_connected_features, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x1 = self.conv_blocks(x)

        x2 = self.pool(x1)
        x2 = x2.view(x2.shape[0], -1)

        x3 = self.linear1(x2)
        x4 = self.dropout(x3)
        x5 = self.linear2(x4)

        return self.softmax(x5)

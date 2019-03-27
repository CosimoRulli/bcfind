# -*- coding: utf-8 -*-
from torchsummary import summary
import torch.nn as nn


class FC_teacher(nn.Module):
    def __init__(self, n_filters, k_conv=3, k_t_conv=3, input_channels=1):

        super(FC_teacher, self).__init__()
        self.input_channels = input_channels
        self.k_conv = k_conv
        self.k_t_conv = k_t_conv
        self.n_filters = n_filters
        # self.batch_size = batch_size
        self.conv1 = nn.Conv3d(input_channels, n_filters, self.k_conv,
                               padding=1)
        # 1x13x13x13 -> n_filters x 13 x 13 x 13
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv3d(n_filters,
                               n_filters * 2, self.k_conv)
        # n_filters x 13 x 13 x 13 -> n_filters*2 x 11 x 11 x 11

        self.conv_t1 = nn.ConvTranspose3d(2*n_filters, n_filters,
                                          self.k_t_conv, padding=1)
        self.conv_t2 = nn.ConvTranspose3d(n_filters,
                                          input_channels, self.k_t_conv)
        # XXX occhio forse è input_size[1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        input = input.unsqueeze(1)

        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.conv_t1(output)
        output = self.relu(output)
        output = self. conv_t2(output)
        output = self.sigmoid(output)

        return output.squeeze(1)


if __name__ == "__main__":
    print('magi')
    teacher = FC_teacher(3, 4, 8, 13).to('cuda:0')
    summary(teacher, input_size=(1, 23, 23, 23))

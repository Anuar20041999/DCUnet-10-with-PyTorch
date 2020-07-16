import torch
from torch.nn.functional import leaky_relu
from torch.nn import Conv1d, ConvTranspose1d


class complex_LeakyReLU(torch.nn.Module):
    def forward(self,input_r, input_i):
        return complex_leaky_relu(input_r, input_i)


class ComplexConv1d(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,input_r, input_i):
        return self.conv_r(input_r)-self.conv_i(input_i), \
               self.conv_r(input_i)+self.conv_i(input_r)


class ComplexConvTranspose1d(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose1d, self).__init__()
        self.conv_tran_r = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)

    def forward(self,input_r,input_i):
        return self.conv_tran_r(input_r)-self.conv_tran_i(input_i), \
               self.conv_tran_r(input_i)+self.conv_tran_i(input_r)

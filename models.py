# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:40:51 2020

@author: Silence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)            
            

class ConvBlock_Down(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, stride=1):        
        super(ConvBlock_Down, self).__init__()
        inter_channels = in_channels//2 if in_channels > out_channels else out_channels//2
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, 
                               out_channels=inter_channels, 
                               kernel_size=3, stride=stride,
                               padding=padding, bias=False)
        
        self.conv2 = nn.Conv3d(in_channels=inter_channels, 
                              out_channels=out_channels,
                               kernel_size=3, stride=stride, 
                               padding=padding, bias=False) 
        
        self.bn1 = nn.BatchNorm3d(inter_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, x, is_pool=True, pool_size=(2, 2, 2), pool_type='avg'):
        
        if is_pool:
            if pool_type == 'max':
                x = F.max_pool3d(x, kernel_size=pool_size, stride=(2, 2, 2))
            elif pool_type == 'avg':
                x = F.avg_pool3d(x, kernel_size=pool_size, stride=(2, 2, 2))
            else:
                raise Exception('Incorrect argument!')
                         
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x

class ConvBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sample=True):        
        super(ConvBlock_Up, self).__init__()
        
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False)
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)       
            
        in_channels = in_channels//2+in_channels
        inter_channels = in_channels//2 if in_channels > out_channels else out_channels//2       
        inter_channels = max(inter_channels, out_channels)
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, 
                               out_channels=inter_channels, 
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        
        self.conv2 = nn.Conv3d(in_channels=inter_channels, 
                              out_channels=out_channels,
                               kernel_size=3, stride=stride, 
                               padding=1, bias=False) 
                          
        self.bn1 = nn.BatchNorm3d(inter_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input1, input2):
        
        x = input1
        x1 = input2
        x = self.sample(x)
        x = torch.cat((x, x1), dim=1) 
        
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = F.relu(self.bn2(self.conv2(x)), inplace=False)
        
        return x

class ConvBlock_Map(nn.Module):
    def __init__(self, in_channels, out_channels):        
        super(ConvBlock_Map, self).__init__()
        inter_channels = in_channels//2 if in_channels > out_channels else out_channels//2
        
        self.conv1 = nn.Conv3d(in_channels=in_channels, 
                               out_channels=inter_channels, 
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.conv2 = nn.Conv3d(in_channels=inter_channels, 
                              out_channels=out_channels,
                               kernel_size=3, stride=1, 
                               padding=1, bias=False) 
        
        self.bn1 = nn.BatchNorm3d(inter_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, x):
                         
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2(self.conv2(x))
        
        return x






class MassNET(nn.Module):
    
    def __init__(self):
        super(MassNET, self).__init__()

        self.conv_block1 = ConvBlock_Down(in_channels=1, out_channels=32)
        self.conv_block_down1 = ConvBlock_Down(in_channels=32, out_channels=64)
        self.conv_block_down2 = ConvBlock_Down(in_channels=64, out_channels=128)
        self.conv_block_down3 = ConvBlock_Down(in_channels=128, out_channels=256)
        
        self.conv_block_up3 = ConvBlock_Up(in_channels=256, out_channels=128)
        self.conv_block_up2 = ConvBlock_Up(in_channels=128, out_channels=64)
        self.conv_block_up1 = ConvBlock_Up(in_channels=64, out_channels=64)
        self.conv_block2 = ConvBlock_Map(in_channels=64, out_channels=1)

    def forward(self, x):
        
        x = x[:, None, :, :, :]
        
        # Encoder
        x1 = self.conv_block1(x, is_pool=False)
        x2 = self.conv_block_down1(x1)
        x3 = self.conv_block_down2(x2)
        out1 = self.conv_block_down3(x3)
        
        # Decoder
        out1 = self.conv_block_up3(out1, x3)
        out1 = self.conv_block_up2(out1, x2)
        out1 = self.conv_block_up1(out1, x1)
        out1 = self.conv_block2(out1)
        
        out1 = out1.mul(1-x)  
        out1 = torch.tanh(out1) 
        
        out1 = out1.squeeze(dim=1)
        
        out2 = torch.mean(out1,(3,2,1))
        out2 = out2*config.scale_Mass/config.dCdX/config.Db
        
        #return output
        return out1, out2

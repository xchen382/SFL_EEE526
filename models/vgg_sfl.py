'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
import torch.nn.functional as F
import torch.nn.init as init
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class AuxClassifier(nn.Module):
    def __init__(self,act_size,num_class):
        super(AuxClassifier, self).__init__()
        self.head = nn.Sequential(
            # nn.Conv2d(act_size[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(act_size[1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_class))
    
    def forward(self, x):
        features = self.head(x)
        return features


class Local(nn.Module):
    def __init__(self,model,num_class):
        super(Local, self).__init__()
        self.head = model
        self.local_classifier = AuxClassifier(self.get_act_size(),num_class)

    def get_act_size(self):
        with torch.no_grad():
           return self(torch.ones((1,3,32,32))).size()

    def forward(self,x):
        return self.head(x)

    def local_forward(self, x):
        output = self.forward(x)
        output = self.local_classifier(output)
        return output


        
class Cloud(nn.Module):
    def __init__(self,model,num_class):
        super(Cloud, self).__init__()
        self.head = model
        classifier_list = [nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True)]

        classifier_list += [nn.Linear(512, num_class)]
        self.classifier = nn.Sequential(*classifier_list)

    def forward(self, x):
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, feature, num_class = 10):
        super(VGG, self).__init__()
        self.local = Local(feature[0],num_class)
        self.cloud = Cloud(feature[1],num_class)

    def forward(self,x):
        output = self.cloud(self.local(x))
        return output


def make_layers(cutting_layer,cfg,norm, batch_norm=False):
    # layer is seperated as client client-bottleneck server-bottleneck server
    local = []
    cloud = []
    in_channels = 3
    #Modified Local part - Experimental feature
    channel_mul = 1
    for v_idx,v in enumerate(cfg):
        if v_idx < cutting_layer - 1:
            if v == 'M':
                local += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, int(v * channel_mul), kernel_size=3, padding=1)
                if batch_norm:
                    if norm == 'batch_norm':
                        local += [conv2d, nn.BatchNorm2d(int(v * channel_mul)), nn.ReLU(inplace=True)]
                    elif norm == 'group_norm':
                        local += [conv2d, nn.GroupNorm(1,int(v * channel_mul)), nn.ReLU(inplace=True)]

                else:
                    local += [conv2d, nn.ReLU(inplace=True)]
                in_channels = int(v * channel_mul)
        elif v_idx == cutting_layer - 1:
            if v == 'M':
                local += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    if norm == 'batch_norm':
                        local += [conv2d, nn.BatchNorm2d(int(v * channel_mul)), nn.ReLU(inplace=True)]
                    elif norm == 'group_norm':
                        local += [conv2d, nn.GroupNorm(1,int(v * channel_mul)), nn.ReLU(inplace=True)]
                else:
                    local += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        else:
            if v == 'M':
                cloud += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    cloud += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    cloud += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

    return nn.Sequential(*local),nn.Sequential(*cloud)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(cutting_layer, num_agent=1,num_class = 10):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cutting_layer,cfg['A'], batch_norm=False),num_agent=num_agent,num_class = num_class)

# def vgg11_bn(cutting_layer, num_agent=1,num_class = 10):
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     return VGG(make_layers(cutting_layer,cfg['A'], batch_norm=True),num_agent=num_agent,num_class = num_class)

def vgg11_bn(norm, cutting_layer, num_classes):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cutting_layer,cfg['A'], norm, batch_norm=True),num_class = num_classes)


def vgg13(cutting_layer, num_agent=1, num_class = 10):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cutting_layer,cfg['B'], batch_norm=False),num_agent=num_agent, num_class = num_class)

def vgg13_bn(cutting_layer, num_agent=1, num_class = 10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cutting_layer,cfg['B'], batch_norm=True),num_agent=num_agent, num_class = num_class)

def vgg11_vib(cutting_layer, num_agent=1, num_class = 10):
    """VGG 11-layer model (configuration "A")"""
    return VGG_vib(make_layers(cutting_layer,cfg['A'], batch_norm=False),num_agent=num_agent,num_class = num_class)

def vgg11_bn_vib(cutting_layer, num_agent=1, num_class = 10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG_vib(make_layers(cutting_layer,cfg['A'], batch_norm=True),num_agent=num_agent, num_class = num_class)

def vgg13_vib(cutting_layer, num_agent=1, num_class = 10):
    """VGG 13-layer model (configuration "B")"""
    return VGG_vib(make_layers(cutting_layer,cfg['B'], batch_norm=False),num_agent=num_agent, num_class = num_class)

def vgg13_bn_vib(cutting_layer, num_agent=1, num_class = 10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG_vib(make_layers(cutting_layer,cfg['B'], batch_norm=True),num_agent=num_agent, num_class = num_class)

def vgg16(cutting_layer, num_agent=1, num_class = 10):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cutting_layer,cfg['D'], batch_norm=False),num_agent=num_agent, num_class = num_class)

def vgg16_bn(cutting_layer, num_agent=1, num_class = 10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cutting_layer,cfg['D'], batch_norm=True),num_agent=num_agent, num_class = num_class)

def vgg19(cutting_layer, num_agent=1, num_class = 10):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cutting_layer,cfg['E'], batch_norm=False),num_agent=num_agent, num_class = num_class)

def vgg19_bn(cutting_layer, num_agent=1, num_class = 10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cutting_layer,cfg['E'], batch_norm=True),num_agent=num_agent, num_class = num_class)
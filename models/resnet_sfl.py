"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,norm ='batch_norm'):
        super().__init__()
        if norm=='batch_norm':
            #residual function
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

            #shortcut
            self.shortcut = nn.Sequential()

            #the shortcut output dimension is not the same with residual function
            #use 1*1 convolution to match the dimension
            if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * BasicBlock.expansion)
                )
        elif norm=='group_norm':
            #residual function
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(1,out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1,out_channels * BasicBlock.expansion)
            )

            #shortcut
            self.shortcut = nn.Sequential()

            #the shortcut output dimension is not the same with residual function
            #use 1*1 convolution to match the dimension
            if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(1,out_channels * BasicBlock.expansion)
                )



    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class AuxClassifier(nn.Module):
    def __init__(self,act_size,num_classes):
        super(AuxClassifier, self).__init__()
        self.head = nn.Sequential(
            # nn.Conv2d(act_size[1], 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(act_size[1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes))
    
    def forward(self, x):
        features = self.head(x)
        return features


class Local(nn.Module):
    def __init__(self, block, cut, num_block, norm, num_classes=100):
        output_channel = [64,128,256,512]
        strides = [1,2,2,2]
        super().__init__()
        self.in_channels = 64
        if norm=='batch_norm':
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                # nn.GroupNorm(1,64),
                nn.ReLU(inplace=True))
        elif norm == 'group_norm':
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1,64),
                nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv_x = []
        #TODO!
        for layer in range(0,cut-1):
            self.conv_x.append(self._make_layer(block, output_channel[layer], num_block[layer], strides[layer], norm))
        self.conv_x = nn.Sequential(*self.conv_x)
        self.local_classifier = AuxClassifier(self.get_act_size(),num_classes=num_classes)

    def get_act_size(self):
        with torch.no_grad():
           return self(torch.ones((1,3,32,32))).size()

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv_x(output)
        return output
    
    def local_forward(self, x):
        output = self.forward(x)
        output = self.local_classifier(output)
        return output



    def _make_layer(self, block, out_channels, num_blocks, stride, norm):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride,norm=norm))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


class Cloud(nn.Module):
    def __init__(self, block, cut, num_block, in_channels,num_classes=100):
        super().__init__()
        output_channel = [64,128,256,512]
        strides = [1,2,2,2]
        self.in_channels = in_channels

        self.conv_x = []
        for layer in range(cut-1,4):
            self.conv_x.append(self._make_layer(block, output_channel[layer], num_block[layer], strides[layer]))
            
        self.conv_x = nn.Sequential(*self.conv_x)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        # output = self.conv3_x(x)
        # output = self.conv4_x(output)
        # output = self.conv5_x(output)
        output = self.conv_x(x)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

class ResNet(nn.Module):

    def __init__(self, block, num_block, cut,num_classes=100,norm='batch_norm'):
        super().__init__()
        if cut >0:
            self.local = Local(block,cut,num_block,norm,num_classes=num_classes)
            self.cloud = Cloud(block,cut,num_block,in_channels=self.local.in_channels,num_classes=num_classes)
        else:
            self.local = None
            self.cloud = Local(block,100,num_block,norm,num_classes=num_classes)

    def forward(self, x):
        output = self.cloud(self.local(x))
        return output

def resnet18(norm,cut,num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],cut,num_classes=num_classes,norm=norm)

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



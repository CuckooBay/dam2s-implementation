import torch
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from typing import Type, Any, Union, List
from collections import OrderedDict 
from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock, Bottleneck

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class IntResNet(ResNet):
    def __init__(self,output_layer,*args):
        self.output_layer = output_layer
        super().__init__(*args)
        
        self._layers = []
        for l in list(self._modules.keys()):
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))

    def _forward_impl(self, x):
        for l in self._layers:
            x = self.layers[l](x)

        m = nn.AdaptiveAvgPool2d((1, 1))
        x = m(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def new_resnet(
    arch: str,
    outlayer: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> IntResNet:

    """
    address to load pretrained model
    """
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    model = IntResNet(outlayer, block, layers, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model


class VFeatureExtractor:
    def __init__(self):
        # self.model = new_resnet('resnet18', 'layer4', BasicBlock, [2, 2, 2, 2], True, True)
        # self.model = new_resnet('resnet34', 'layer4', BasicBlock, [3, 4, 6, 3], True, True)
        self.model = new_resnet('resnet50', 'layer4', Bottleneck, [3, 4, 6, 3], True, True)

    def __call__(self, images):
        '''
            images.shape = (n, 224, 224, 3)
        '''
        images = self.reshape_images(images) # output: (n, 3, 224, 224)
        fds = self.model(images).cpu().data.numpy()
        fds = np.array([ fd.flatten() for fd in fds ])
        return fds

    def reshape_images(self, images):
        # change from (n, 224, 224, 3) to (n, 3, 224, 224)
        images = torch.from_numpy(np.transpose(images, (0, 3, 1, 2)))
        images = images.float()
        return images

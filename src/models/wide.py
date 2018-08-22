
'''
The Wide-Net described in our paper.
'''

import torch
import torch.nn as nn
import math
import numpy as np

cfg = {
   'wide': [128, 'M', 256, 'M',128, 64],
}

class multi_pool(nn.Module):
	def __init__(self):
		super(multi_pool, self).__init__()
		self.pool2 = nn.MaxPool2d(2, stride=2)
		self.pool4 = nn.MaxPool2d(4, stride=2, padding=1)
		self.pool8 = nn.MaxPool2d(8, stride=2, padding=3)         
	def forward(self, x):
		x1 = self.pool2(x)
		x2 = self.pool4(x)
		x3 = self.pool8(x)
		y = (x1+x2+x3)/3.0
		return y
    
class stack_pool(nn.Module):
	def __init__(self):
		super(stack_pool, self).__init__()
		self.pool2 = nn.MaxPool2d(2, stride=2)
		self.pool2s1 = nn.MaxPool2d(2, stride=1) 
		self.pool3s1 = nn.MaxPool2d(3, stride=1, padding=1)       
		self.padding = nn.ReplicationPad2d((0, 1, 0, 1))        
	def forward(self, x):
		x1 = self.pool2(x) 
		x2 = self.pool2s1(self.padding(x1))
		x3 = self.pool3s1(x2)
		y = (x1+x2+x3)/3.0
		return y
    
class feature_net(nn.Module):
	def __init__(self,pool):
		super(feature_net, self).__init__()
		self.pool = pool
		self.features = self.make_layers(cfg = cfg['wide'], batch_norm = False)
	def forward(self, x):
		feature = self.features(x)
		return feature
	def make_layers(self, cfg, batch_norm = False):
		layers = []
		in_channels = 1
		idx_M = 0 
		conv_size = 7        
		for v in cfg:
			if v == 'M':
				idx_M += 1
				if idx_M >= 1:
					conv_size = 5       
				if idx_M >= 2:
					conv_size = 3  
				if self.pool == 'mpool':
					layers += [multi_pool()]
				if self.pool == 'stackpool':
					layers += [stack_pool()]        
				if self.pool == 'vpool':
					layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size = conv_size, padding = (conv_size-1)/2 )
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
				else:
					layers += [conv2d, nn.ReLU(inplace = True)]
				in_channels = v
		return nn.Sequential(*layers)

class wide(nn.Module):
	def __init__(self,pool):
		super(wide, self).__init__()
		self.conv2d = nn.Conv2d(64, 1, kernel_size = 1)
		self.feature_net = feature_net(pool)
		#self._initialize_weights()
	def forward(self, x):
		x = self.feature_net.forward(x)
		heat_map = self.conv2d(x)
		return heat_map
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
                   
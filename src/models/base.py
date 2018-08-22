
'''
The Base-Net described in our paper.
'''

import torch
import torch.nn as nn
from src.network import Conv2d
import time

class base(nn.Module):
    
    def __init__(self, pool, bn=False):
        super(base, self).__init__()
        
        kernel_size = 5
        self.pool = pool
        if kernel_size==7:
            self.c1 = Conv2d( 1, 16, 9, same_padding=True, bn=bn)
            self.c2 = Conv2d(16, 32, 7, same_padding=True, bn=bn)
            self.c3_5 = nn.Sequential(Conv2d(32, 16, 7, same_padding=True, bn=bn),
                             Conv2d(16,  8, 7, same_padding=True, bn=bn),
                             Conv2d( 8, 1, 1, same_padding=True, bn=bn))            
        if kernel_size==5:
            self.c1 = Conv2d( 1, 20, 7, same_padding=True, bn=bn)
            self.c2 = Conv2d(20, 40, 5, same_padding=True, bn=bn)
            self.c3_5 = nn.Sequential(Conv2d(40, 20, 5, same_padding=True, bn=bn),
                             Conv2d(20, 10, 5, same_padding=True, bn=bn),
                             Conv2d( 10, 1, 1, same_padding=True, bn=bn))
        if kernel_size==3:
            self.c1 = Conv2d( 1, 24, 5, same_padding=True, bn=bn)
            self.c2 = Conv2d(24, 48, 3, same_padding=True, bn=bn)
            self.c3_5 = nn.Sequential(Conv2d(48, 24, 3, same_padding=True, bn=bn),
                             Conv2d(24, 12, 3, same_padding=True, bn=bn),
                             Conv2d( 12, 1, 1, same_padding=True, bn=bn))
                                  
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool2s1 = nn.MaxPool2d(2, stride=1) 
        self.pool3s1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(4, stride=2, padding=1)
        self.pool8 = nn.MaxPool2d(8, stride=2, padding=3)
        
        self.padding = nn.ReplicationPad2d((0, 1, 0, 1))        
       
    def multi_pool(self, x):
        x1 = self.pool2(x)
        x2 = self.pool4(x)
        x3 = self.pool8(x) 
        y = (x1+x2+x3)/3.0        
        return y 
    
    def stack_pool(self, x):
        x1 = self.pool2(x) 
        x2 = self.pool2s1(self.padding(x1))
        x3 = self.pool3s1(x2)
        y = (x1+x2+x3)/3.0
        return y 
       
    def forward(self, im_data):         
        x = self.c1(im_data)      

        if self.pool=='mpool':
            x = self.multi_pool(x) 
        if self.pool=='stackpool':
            x = self.stack_pool(x) 
        if self.pool=='vpool':
            x = self.pool2(x)

        x = self.c2(x)
        
        if self.pool=='mpool':
            x = self.multi_pool(x) 
        if self.pool=='stackpool':
            x = self.stack_pool(x) 
        if self.pool=='vpool':
            x = self.pool2(x)
            
        x = self.c3_5(x)
            
        return x
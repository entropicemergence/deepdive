# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 06:07:59 2023


@author: gesit
"""

import cupy as cp


class Activation:
    def __init__(self, config):
        self.batch_size=config.batch_size
        self.mask=[]
    def ReLu(self, input_data):
#         print ("in",input_data.shape)
        self.mask=input_data>0
        return input_data*self.mask
    def backward(self, error):
#         print (self.mask.shape, error.shape)
#         plt.plot((error[0,0]*self.mask[0, 0]).get())
        return self.mask*error
    def update(self):
        a=0
        


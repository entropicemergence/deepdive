# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 06:05:20 2023

@author: gesit
"""

import cupy as cp

class Cross_Entropy:
    def __init__(self,config):
        self.batch_size=config.batch_size
        self.unique_out_token=config.unique_out_token
        self.max_input_length=config.max_input_length
        
        self.index=cp.arange(self.batch_size*self.max_input_length).astype(cp.int32)
        
    def forward(self, softmax_output, target):     
#         for j in range (32):
#             softmax_output[j,cp.arange(target.shape[1]),target[j].astype(cp.int32)]=0.9
# #         print (softmax_output.shape, target.shape)
        # print (softmax_output.shape, target.shape )
        # mask=((target!=0)*1.0).astype(cp.float32)
        # target=target.astype(cp.int32)
        
        # print (target.shape)
        
        mask=(target!=0)
        
        # print (type(mask))
        # print (mask[0])
        
        softmax_output=softmax_output.reshape([self.batch_size*softmax_output.shape[1],-1])
        
        
        target=cp.array(target.reshape([softmax_output.shape[0]])).astype(cp.int32)
        
        a=softmax_output[self.index[:target.shape[0]],target]
        
        mask=mask.reshape([softmax_output.shape[0]])
        mask_a=cp.array(mask*1.0).astype(cp.int32)
        mask_b=cp.array(~mask*1.0).astype(cp.int32)
        
        
        a=a*mask_a+mask_b
        
        # print (a.shape, mask.shape, cp.dtype(a), a[0:30])
        
        cross_entropy_loss=cp.mean(-cp.log(a))
        
        # error=-((1.0/a)/self.unique_out_token)
        
        error=-(1.0/a)
        
        # print (error[:30])
        
        error=error*mask_a
        
        # print (error[:30])
        
        # print (error.shape, softmax_output.shape, target.shape )
        
        return cross_entropy_loss, error, target
    
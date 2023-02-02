# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 06:16:18 2023

@author: gesit
"""

import cupy as cp


class Adam:
    def __init__(self, optimized_parameter, alpha=0.001, beta=0.9, beta2= 0.999, epsilon=1e-10, param_dtype=cp.float32):
        self.alpha = alpha
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m=cp.zeros(optimized_parameter.shape).astype(param_dtype)
        self.m_diff, self.v, self.v_diff =self.m.copy(), self.m.copy(), self.m.copy() 
        
        self.Iteration=1
        
    def optimize(self, optimized_parameter, grad):
        
        self.m = self.beta*self.m + (1-self.beta)*grad
        
        self.v = self.beta2*self.v+ (1-self.beta2)*(grad**2)
        
        self.m_diff =self.m / (1-(self.beta**self.Iteration))
        
        self.v_diff=self.v / (1-(self.beta2**self.Iteration))
        
        # print ("sum",cp.sum(self.w),  cp.sum(self.m), cp.sum(self.v), cp.sum(self.m_diff), cp.sum(self.v_diff), cp.sum(DW))
        
        self.Iteration+=1
        
        return optimized_parameter-((self.alpha*self.m_diff)/(cp.sqrt(self.v_diff)+self.epsilon))
    
        
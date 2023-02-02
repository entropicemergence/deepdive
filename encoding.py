# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 06:13:07 2023

@author: gesit
"""

import cupy as cp
from layer import Dense


class positional_encoding:
    
    def __init__(self, config):
        self.batch_size=config.batch_size
        self.opening_width=config.opening_width
        self.dtype=config.dtype
        self.max_input_length=config.max_input_length
        
        self.scale=cp.sqrt(self.opening_width).astype(self.dtype)
        
        c=((10000**(-cp.arange(0,self.opening_width,2)/self.opening_width))*cp.arange(0,self.max_input_length).reshape([-1,1])).astype(self.dtype)
        self.pe=cp.zeros([self.max_input_length, self.opening_width]).astype(self.dtype)            #ensure it has the right data type
        self.pe[:,0::2]=cp.sin(c)
        self.pe[:,1::2]=cp.cos(c)
#         plt.figure(figsize=(20,20))
#         plt.imshow(self.positional_embedding[:,0::2])        


    def forward (self, input_data):
#         print (input_data.shape, self.pe.shape)
#         print (a.shape)
#         plt.imshow(a[0,:,:])
#         plt.show()
#         plt.plot(a[0,10,0::2])
        return input_data+self.pe[:input_data.shape[1]]
    def backward(self,error):
        return (error)
    


class word_embedding:
    def __init__(self,config, encoder_decoder):
#         print (config)
        self.batch_size=config.batch_size
        self.opening_width=config.opening_width
        if encoder_decoder:
            self.unique_in_token=config.unique_in_token
        else:
            self.unique_in_token=config.unique_out_token
        self.dtype=config.dtype
#         self.w=cp.random.normal(0, self.unique_in_token**-0.5, (self.unique_in_token,self.opening_width)).astype(self.dtype)
        self.one_hot_vector=[]
        self.compacted_index=[]
        
        self.linear=Dense(config, self.unique_in_token,self.opening_width)
        self.dtype=config.dtype
        
    def transform_input_to_fit(self,input_data, training=True):
#         print (input_data.shape)#[batch size , input token] > [batch size, input token, vocabulary token ]
        if training:
            # self.batch_size=input_data.shape[0]
            batch_size=self.batch_size
        else:
            batch_size=input_data.shape[0]

        # print (type(input_data))
        a=cp.max(input_data,axis=0)
        # print (type(input_data),type(a))
        
        
        for i, a in enumerate (cp.max(input_data,axis=0),0):
            if a==0 and i >0:
                break
        self.compacted_index= input_data[:,:i+1]
        input_token=self.compacted_index.shape[1]
       
        output =cp.zeros([int(batch_size)*input_token, int(self.unique_in_token)]).astype(self.dtype)
        # print (cp.dtype(output), (input_data.dtype))
        output[cp.arange((int(batch_size))*input_token),self.compacted_index.reshape([1,-1]).astype(cp.int32)]=1
        
        # print (type(output))
        return output.reshape([int(batch_size),input_token,-1])
    
    

    def forward(self, input_data, training=True):
        # print (input_data.dtype)
        
        self.one_hot_vector=self.transform_input_to_fit(input_data, training)
        
#         print (input_data.shape, self.w.shape)
#         output_data=cp.dot(self.one_hot_vector, self.w)
        
        # cp.set_printoptions(suppress=True, precision=2, linewidth=1000)
        
        # print (self.one_hot_vector.shape, cp.argmax(self.one_hot_vector,2), cp.max(self.one_hot_vector,2), cp.sum(self.one_hot_vector,2))
        
        # print (cp.dtype(self.one_hot_vector))
        return self.linear.forward(self.one_hot_vector)
        # return self.linear.forward(cp.array(self.one_hot_vector)) #slower
    
    def backward(self,error):
#         print (error.shape,self.one_hot_vector.shape)
        return self.linear.backward(error)

    def update_param(self):    
        self.linear.update_param()
        
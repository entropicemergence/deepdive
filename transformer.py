# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 06:05:40 2023

@author: gesit
"""

import cupy as cp
from activation import Activation
from layer import Dense, Softmax
from norm import Add_Norm






class Pure_Decoder_Layer:
    def __init__(self,config):
        
        self.batch_size=config.batch_size
        self.width=config.opening_width
        self.dtype=config.dtype
        self.head_count=config.decoder_heads        
        self.head_width=config.head_width
        self.feed_forward_width=config.feed_forward_width
        
        
        self.masked_multihead_attention=Multihead_Attention(config, self.head_count, self.head_width, self.feed_forward_width, self.width, mask_bool=True)
        
        self.add_norm1=Add_Norm(config, self.width)
        
        self.feed_forward_1=Dense(config, self.width, self.feed_forward_width)
        self.activation=Activation(config)
        self.feed_forward_2=Dense(config, self.feed_forward_width, self.width)
        
        self.add_norm2=Add_Norm(config, self.width)
           
        
    def forward(self, prev_layer_o, mask, training=True):
        
        out=self.masked_multihead_attention.forward(prev_layer_o,prev_layer_o,prev_layer_o, mask, training)
        
        out=self.add_norm1.forward(out, prev_layer_o)
        
        ff_out=self.feed_forward_1.forward(out)
        ff_out=self.activation.ReLu(ff_out)
        ff_out=self.feed_forward_2.forward(ff_out)
        
        ff_out=self.add_norm2.forward(ff_out, out)
        
        return ff_out
    
    def backward(self, error):
        
        error_split=self.add_norm2.backward(error)
        
        error=self.feed_forward_2.backward(error_split)
        error=self.activation.backward(error)
        error=self.feed_forward_1.backward(error)
        
        error=self.add_norm1.backward(error+error_split)
        
        a,b,c=self.masked_multihead_attention.backward(error)
        
        return (a+b+c)+error
        
        
    def update_param(self):        
        self.masked_multihead_attention.update_param()
        self.add_norm1.update_param()
        self.feed_forward_1.update_param()
        self.feed_forward_2.update_param()
        self.add_norm2.update_param()


class Decoder_Layer:
    def __init__(self,config):
        
        self.batch_size=config.batch_size
        self.width=config.opening_width
        self.dtype=config.dtype
        self.head_count=config.decoder_heads        
        self.head_width=config.head_width
        self.feed_forward_width=config.feed_forward_width
        
        self.masked_multihead_attention=Multihead_Attention(config, self.head_count, self.head_width, self.feed_forward_width, self.width, mask_bool=True)
        self.sideload_multihead_attention=Multihead_Attention(config, self.head_count, self.head_width, self.feed_forward_width, self.width)
        
        self.feed_forward_1=Dense(config, self.width, self.feed_forward_width)
        self.activation=Activation(config)
        self.feed_forward_2=Dense(config, self.feed_forward_width, self.width)
        
        self.add_norm=Add_Norm(config, self.width)
        
        
    def forward(self, from_enc, prev_layer_o, mask, training=True):
        
        out=self.masked_multihead_attention.forward(prev_layer_o,prev_layer_o,prev_layer_o, mask[0])
        
#         print (from_enc.shape,out.shape)
        out=self.sideload_multihead_attention.forward(out, from_enc, from_enc, mask[1]) #input order (q,k,v,mask)
        
        # What comes out of the encoder should be the key and value matrix 
        # and what comes out of the bottom part of the decoder is the query matrix."
        
        ff_out=self.feed_forward_1.forward(out)
        ff_out=self.activation.ReLu(ff_out)
        ff_out=self.feed_forward_2.forward(ff_out)
        
        ff_out=self.add_norm.forward(ff_out,out)
        
        return ff_out
    def backward(self, error):
        error_split=self.add_norm.backward(error)
        
        error1=self.feed_forward_2.backward(error_split)
        error1=self.activation.backward(error1)
        error1=self.feed_forward_1.backward(error1)
        
        errorq, errork_side, errorv_side=self.sideload_multihead_attention.backward(error1+error_split)
        
        a,b,c=self.masked_multihead_attention.backward(errorq)
        
        return (a+b+c), errork_side + errorv_side
        
        
    def update_param(self):        
        self.masked_multihead_attention.update_param()

        self.sideload_multihead_attention.update_param()

        self.feed_forward_1.update_param()
        
        self.feed_forward_2.update_param()
        
        self.add_norm.update_param()
        
        a=0
        
        
        
        
class Encoder_Layer:
    def __init__(self,config):
        self.batch_size=config.batch_size
        self.width=config.opening_width
        self.dtype=config.dtype
        self.head_count=config.encoder_heads        
        self.head_width=config.head_width
        self.feed_forward_width=config.feed_forward_width
        
        self.multihead_attention=Multihead_Attention(config, self.head_count, self.head_width, self.feed_forward_width, self.width  )
        
        self.feed_forward_1=Dense(config, self.width, self.feed_forward_width)
        self.activation=Activation(config)
        self.feed_forward_2=Dense(config, self.feed_forward_width, self.width)
        
        self.add_norm=Add_Norm(config, self.width)
        
    def forward(self, input_data, mask, training=True):

        out=self.multihead_attention.forward(input_data,input_data,input_data, mask)
        
        
        ff_out=self.feed_forward_1.forward(out)
        ff_out=self.activation.ReLu(ff_out)
        ff_out=self.feed_forward_2.forward(ff_out)
        
        ff_out=self.add_norm.forward(ff_out,out)
        # print (cp.dtype(ff_out))
#         print (out.shape,ff_out.shape)
        return ff_out

    def backward(self, error):
        
        error_split=self.add_norm.backward(error)
        
        error1=self.feed_forward_2.backward(error_split)
        error1=self.activation.backward(error1)
        error1=self.feed_forward_1.backward(error1)
        
        a,b,c=self.multihead_attention.backward(error1+error_split)
        
        return a+b+c
    def update_param(self):
        self.multihead_attention.update_param()
        self.feed_forward_1.update_param()
        self.feed_forward_2.update_param()
        self.add_norm.update_param()
        
        
class Multihead_Attention:
    def __init__(self,config, head_count, head_width, feed_forward_width, opening_width, mask_bool=False):
        # mask=cp.ones([50,50])+cp.arange(50)
        # mask=((mask+cp.flip(mask.T))>51)*1.0
        # mask[mask==1]=-cp.inf
        # self.mask=mask.reshape([1,1,50,50])
        # self.mask=config.mask
        
        
        self.mask_bool=mask_bool
        self.batch_size=config.batch_size
        self.dtype=config.dtype
        
        
        self.width=opening_width
        self.head_count=head_count       
        self.head_width=head_width
        self.feed_forward_width=feed_forward_width
        
        
        self.Q=Dense(config, self.width, self.head_width*self.head_count) #multi head calculated as one (width*num heads)
        self.K=Dense(config, self.width, self.head_width*self.head_count) #multi head calculated as one (width*num heads)
        self.V=Dense(config, self.width, self.head_width*self.head_count) #multi head calculated as one (width*num heads)
        self.softmax=Softmax(config)
        self.linear=Dense(config, self.head_width*self.head_count, self.width) #linear layer after attention heads, 
        self.add_norm=Add_Norm(config, self.width)
        
        self.attention_dim=cp.sqrt(self.width/self.head_count).astype(self.dtype)

        
        self.q=[]
        self.k=[]
        self.v=[]
        self.qk=[]

        self.mask_b=[]
        
        
    def forward(self,q_in,k_in,v_in, mask, training=True):
        
        if training:
            batch_size=self.batch_size
        else:
            batch_size=q_in.shape[0]
            
        # print (batch_size)
        self.q=self.Q.forward(q_in)
        self.k=self.K.forward(k_in)
        self.v=self.V.forward(v_in)
        
        # self.attention_dim=cp.sqrt(self.q.shape[-2]).astype(self.dtype)

        self.q=self.q.reshape([batch_size, -1, self.head_count, self.head_width]) #numpy.moveaxis(a, source, destination)
        self.k=self.k.reshape([batch_size, -1, self.head_count, self.head_width])
        self.v=self.v.reshape([batch_size, -1, self.head_count, self.head_width])


        self.q,self.k,self.v=cp.moveaxis(self.q,1,2),cp.moveaxis(self.k,1,2),cp.moveaxis(self.v,1,2)
#         print (self.q.shape,self.k.shape,self.v.shape)
        

        self.qk=cp.matmul(self.q, self.k.transpose(0,1,3,2))/(self.attention_dim)              #q k matmul then divided by dimension(scalling)
        
        # if self.mask_bool:
        #     n=self.qk.shape[-1]
        #     self.mask_b=self.mask[:,:,:n,:n]
        #     self.qk=self.qk+self.mask_b
        
        
        self.mask_b=mask[:,:,:self.qk.shape[2],:self.qk.shape[3]]
        self.qk=self.qk+self.mask_b
        
        
        # print (self.qk[0,0])
        # print (self.qk.shape, mask.shape)
            

            
            
        #     print (self.mask_bool, self.qk.shape, self.mask.shape)
        #     self.qk=self.softmax.forward(self.qk) 
        #     print (self.qk[0,0]) 
        # else:
        #     self.qk=self.softmax.forward(self.qk) 

        # cp.set_printoptions(suppress=True, precision=2, linewidth=1000)
#         print (self.qk.shape,self.q.shape, self.k.shape) #(32, 8, 21, 21) (32, 8, 21, 48) (32, 8, 48, 21)
        self.qk=self.softmax.forward(self.qk)  
                            #softmamx to maximize promiinent feature
        qkv=cp.matmul(self.qk,self.v)#(32, 8, 21, 48) (32, 8, 21, 21) (32, 8, 21, 48)# Value vector flitered by qk matrix
        
        # print (qkv[0,0,-3:])
        # print (self.qk[0,0,0::2,0::2])
        # print (self.v[0,0,-3:])
        
        qkv=cp.moveaxis(qkv,1,2).reshape(batch_size,-1,self.head_count*self.head_width) #all heads concatennated

        out=self.linear.forward(qkv)

        out=self.add_norm.forward(out,q_in)
        
        return out   
    
    def backward(self,error):
        split_error=self.add_norm.backward(error)
    
        error1=self.linear.backward(split_error)
        error1=cp.moveaxis(error1.reshape(self.batch_size, -1, self.head_count, self.head_width),1,2)

        errorv=cp.matmul(cp.transpose(self.qk,(0,1,3,2)), error1)
        errorqk=cp.matmul(error1, cp.transpose(self.v,(0,1,3,2)))
        
        # errorqk=errorqk.reshape([self.batch_size,self.head_count,-1])
        errorqk=self.softmax.backward(errorqk)
        # errorqk=errorqk.reshape(self.batch_size, self.head_count, -1 , self.qk.shape[-1])
        
        # cp.set_printoptions(suppress=True)
        
        # print (errorqk[0])
        errorqk=cp.where(self.mask_b == -cp.inf, 0, errorqk)#float("-1e20")
        # print (errorqk[0])
        
        
        errorqk=errorqk/self.attention_dim
        
        # if self.mask_bool:
            # errorqk=errorqk+self.mask_b
            # print (errorqk[0,0])


        # errorq=cp.matmul(errorqk, cp.transpose(self.k,(0,1,3,2)))
        # errork=cp.matmul(cp.transpose(self.q,(0,1,3,2)), errorqk)
        # errorq,errork,errorv=cp.moveaxis(errorq,1,2),cp.transpose(errork,(0,3,1,2)),cp.moveaxis(errorv,1,2)
        # shape=(self.batch_size, -1, self.head_count* self.head_width)
        # errorq,errork,errorv=errorq.reshape(shape),errork.reshape(shape),errorv.reshape(shape)
        # print (self.k.shape, errorqk.shape)
        # print(self.k.shape, self.q.shape, self.v.shape, errorqk.shape)
        
        errorq=cp.matmul(errorqk, self.k)
        errork=cp.matmul(cp.transpose(self.q,(0,1,3,2)), errorqk)
        errorq,errork,errorv=cp.moveaxis(errorq,1,2),cp.transpose(errork,(0,3,1,2)),cp.moveaxis(errorv,1,2)
        
        shape=(self.batch_size, -1, self.head_count* self.head_width)
        errorq,errork,errorv=errorq.reshape(shape),errork.reshape(shape),errorv.reshape(shape)
        
        errorq=self.Q.backward(errorq)
        errork=self.K.backward(errork)
        errorv=self.V.backward(errorv)
        
#         print (errorq.shape, errork.shape, errorv.shape, split_error.shape)
#         print ("error",error1.shape, errorv.shape, errorqk.shape)
        # return error,error,error
        return errorq+split_error, errork, errorv

    def update_param(self):    
        self.Q.update_param()
        self.K.update_param()
        self.V.update_param()
        a=0
        self.linear.update_param()
        self.add_norm.update_param()
        

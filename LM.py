# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 08:27:13 2023

@author: gesit
"""



# import matplotlib.pyplot as plt
# import time
import cupy as cp
import numpy as np
# import pickle as pkl
from tqdm import tqdm

from data import CN, text_process, file_process, mask

from layer import Softmax,  Dense, Dropout
from encoding import positional_encoding, word_embedding
from transformer import Pure_Decoder_Layer
from loss import Cross_Entropy

cp.set_printoptions(suppress=True, precision=1, linewidth=1000)
# import sys




def config():
    C=CN()
    C.batch_size=16
    C.max_input_length=128
    C.sequence_overlap=10
    
    C.opening_width=256
    C.decoder_layer=3
    C.decoder_heads=8
    
    C.head_width=32
    C.feed_forward_width=512
    
    # C.unique_in_token=5960
    # C.unique_out_token=7802
    C.unique_in_token=4096
    C.unique_out_token=4096
    
    
    C.dtype=cp.float32
    C.dttype2=cp.int32
    return C



class LM:
    def __init__(self,config):
        
        self.unique_out_token=config.unique_out_token
        self.width=config.opening_width
        self.dtype=config.dtype
        self.scale=cp.sqrt(self.width).astype(self.dtype)
        self.decoder_layer_num=config.decoder_layer
        
        self.create_layers(config)
        
    def create_layers(self, config):
        
        self.word_embedding=word_embedding(config, encoder_decoder=False) #encoder_decoder False means using output vocabulary
        self.positional_encoding=positional_encoding(config)
        self.dropout=Dropout(config)
        
        
        self.decoder_layers=[]
        for j in range(self.decoder_layer_num):
            self.decoder_layers.append(Pure_Decoder_Layer(config))
            
        self.final_linear=Dense(config,self.width, self.unique_out_token)
        self.softmax_final_1D=Softmax(config)
         
    def forward(self, prev_out, mask ,training=True):

        a=self.word_embedding.forward(prev_out, training)*self.scale

        a=self.positional_encoding.forward(a)

        a=self.dropout.forward(a,training)
        
        for decoder_layer in self.decoder_layers:
            a=decoder_layer.forward(a, mask, training)

        a=self.final_linear.forward(a)
       
        a=self.softmax_final_1D.softmax1D(a, self.word_embedding.compacted_index)  #self.word_embedding.compacted_index exixt to debug
        
        return a, self.word_embedding.compacted_index


    def backward(self, error,target):

        error=self.softmax_final_1D.softmax1D_back(error, target)
        error=self.final_linear.backward(error)
        
        for decoder_layer in reversed(self.decoder_layers):
            error=decoder_layer.backward(error)

        error=self.dropout.backward(error)
        error=self.positional_encoding.backward(error)*self.scale
        error=self.word_embedding.backward(error)


    def update_param(self):
        self.word_embedding.update_param()
        j=0
        for decoder_layer in self.decoder_layers:
            if j==0:
                decoder_layer.update_param()
            if j==1:
                decoder_layer.update_param()
            if j==2:
                decoder_layer.update_param()
            j+=1
            
        self.final_linear.update_param()
        
    def imagine(self, initial_seq, max_token):
        
        M=mask()

        m=M.diagonal(initial_seq.shape[-1])
        
        result=initial_seq[0]
        for j in range(max_token):
            m=M.diagonal(initial_seq.shape[-1])
            a, _ = self.forward(initial_seq, m, training=False)
            result = np.append (result, np.argmax(a[0],1)[-1])
            initial_seq=result[-128:].reshape([1,-1])
            
        # print ("sa")
        # print (initial_seq)
        # print (cp.argmax(a[0],1))
        return np.array(result.get()).astype(np.int32)
        
        
        
C=config()
F=file_process()
M=mask()

# T=F.load_gzip("tokenizer")
# encoded_text=F.load_gzip("encoded_train_data")



T=F.load_gzip("oregairu_the_end_of_affair_T")
encoded_text=F.load_gzip("oregairu_the_end_of_affair_encoded")



# lm=LM(C)
lm=F.load_gzip("oregairu_the_end_of_affair")


loss=Cross_Entropy(C)

seq_length=C.max_input_length

maskk=M.diagonal(C.max_input_length)

k=len(encoded_text.ids)//(seq_length*C.batch_size)

qq=np.arange(k)










# print (sys.argv)

# for batch_num in range (10):
#     J=tqdm(qq)
#     for j in J:
#         start=(j*(seq_length+1))*C.batch_size
#         end=start+((seq_length+1)*C.batch_size)
#         text_a=np.array(encoded_text.ids[start:end]).reshape([C.batch_size,-1])
        
#         # print (text_a[:,0:-1].shape,text_a[:,1:].shape )
        
#         output_data, _ =  lm.forward(text_a[:,0:-1], maskk, training=True)
        
#         loss_value, error, target = loss.forward(output_data, text_a[:,1:])
        
#         error_1  = lm.backward(error,target)
        
#         lm.update_param()
#         ac=cp.argmax(output_data, axis=2)
#         ac=np.array(ac.get())
#         target_a=np.array(target.reshape(ac.shape).get()).astype(np.int32)
#         mask_a=(target_a!=0)*1
#         mask_b=(ac==target_a)*mask_a      
#         J.set_description(f"Loss {loss_value:.4f} | Precentage Right {np.sum(mask_b)/np.sum(mask_a):.4f} | Batch {(batch_num*k)+j}")








prompt='his middle . he talks in statistics , only holds opinions on the weather and the current state of government , and now'


prompt=T.encode(prompt)
prompt=np.array(prompt.ids).reshape([1,-1]).astype(np.int32)


# print ("pr", prompt)
result=lm.imagine(prompt, 1000)

# print (type(result))
print (T.decode(result))
# print (prompt.ids)
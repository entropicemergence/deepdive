# import matplotlib.pyplot as plt
import time

import cupy as cp
import numpy as np
# np.set_printoptions(suppress=True)
import pickle as pkl


from data import CN, text_process

from layer import Softmax,  Dense, Dropout
from encoding import positional_encoding, word_embedding
from transformer import Decoder_Layer, Encoder_Layer
from loss import Cross_Entropy

    

def config():
    C=CN()
    C.batch_size=32
    C.max_input_length=51
    C.opening_width=256
    # C.opening_width=128
    C.encoder_layer=3
    C.decoder_layer=3
    C.encoder_heads=8
    C.decoder_heads=8
#     C.head_width=C.opening_width//C.encoder_heads
    C.head_width=32
    C.feed_forward_width=512
    # C.feed_forward_width=256
    
    C.dropout=0.1
    C.unique_in_token=5960
    C.unique_out_token=7802
    C.dtype=cp.float32
    C.dttype2=cp.int32
    
    # mask=mask=np.ones([50,50])+np.arange(50)
    # C.mask=((mask+np.flip(mask.T))<=51)*1
    C.add_norm=False
    C.counter=0

    return C

# text_process=text_process("dataset")
# aaa=text_process.prepare_data()
# pickle_decoder = open(f'text_process3.pkl', 'wb')
# pkl.dump(aaa, pickle_decoder)
# pickle_decoder.close()


pickle_decoder = open(f'text_process3.pkl', 'rb')
# pickle_decoder = open(f'text_process.pkl', 'rb')
aaa=pkl.load(pickle_decoder)
pickle_decoder.close()


transformer=CN()
transformer.Data=CN()

transformer.Data=aaa

transformer.Model=CN()
# transformer.Model.config=config(transformer.Data)
transformer.Model.config=config()


        






class Decoder:
    def __init__(self,config):
        # self.decoder_mask=config.mask
        self.unique_out_token=config.unique_out_token
        self.width=config.opening_width
        self.dtype=config.dtype
        
        self.scale=cp.sqrt(self.width).astype(self.dtype)
        
        self.word_embedding=word_embedding(config, encoder_decoder=False) #encoder_decoder False means using output vocabulary
        self.positional_encoding=positional_encoding(config)
        self.dropout=Dropout(config)
        self.decoder_layer_num=config.decoder_layer
        
        self.decoder_layers=[]
        for j in range(self.decoder_layer_num):
            self.decoder_layers.append(Decoder_Layer(config))
            
        self.final_linear=Dense(config,self.width, self.unique_out_token)
        self.softmax_final_1D=Softmax(config)
        # self.side_error=[]
        
        # self.debug_variable=cp.random.random((32, 40, 256))
        
        
        
        
         
    def forward(self, from_enc, prev_out, mask ,training=True):
        
        a=self.word_embedding.forward(prev_out)*self.scale
        
        a=self.positional_encoding.forward(a)
        a=self.dropout.forward(a,training)
        
        # print (a.shape)
        # a=self.debug_variable
        
        for decoder_layer in self.decoder_layers:
            a=decoder_layer.forward(from_enc,a, mask)
        
        
        # print (a.shape)
        # a=self.final_linear.forward(self.debug_variable)
        
        a=self.final_linear.forward(a)
        a=self.softmax_final_1D.softmax1D(a, self.word_embedding.compacted_index)  #self.word_embedding.compacted_index eexixt to debug
        
        return a, self.word_embedding.compacted_index


    def backward(self, error,target):
#         print (error.shape, target.shape)
        # cp.set_printoptions(suppress=True, precision=10, linewidth=1000)
        # print (cp.min(error,2)[:3])
        
        error=self.softmax_final_1D.softmax1D_back(error, target)

        # print (cp.sum(error,2))
        
        error=self.final_linear.backward(error)
        
        
        # print (cp.sum(error,2))
        
        
        j=0  
        side_error=[]
        for decoder_layer in reversed(self.decoder_layers):
            if j==0:
                error, side_error=decoder_layer.backward(error)
            else :
                error, side_e=decoder_layer.backward(error)
                side_error=side_error+side_e
            # print (cp.max(error,2)[:3,0::2])
            j+=1
            
        # print ("error                                    ",cp.max(error), cp.mean(error))
        # print (cp.sum(error,2)[:,0::2])
        error=self.dropout.backward(error)
        
        # print (cp.sum(error,2))
        
        error=self.positional_encoding.backward(error)*self.scale
        
        # print (cp.sum(error,2)[:,0::2])
        error=self.word_embedding.backward(error)
        
        # side_error=error
        return error, side_error
    
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
        
        

    
class Encoder:
    def __init__(self,config):
#         super(encoder, self).__init__()
        self.encoder_layer_num=config.encoder_layer
        
        self.width=config.opening_width
        self.dtype=config.dtype
        
        self.scale=cp.sqrt(self.width).astype(self.dtype)
    
        self.word_embedding=word_embedding(config, encoder_decoder=True) #encoder_decoder True means using input vocabulary
        self.positional_encoding=positional_encoding(config)
        self.dropout=Dropout(config)
        
        self.encoder_layers=[]
        for j in range (self.encoder_layer_num):
            self.encoder_layers.append(Encoder_Layer(config))

    def forward(self,input_data, mask ,training=True):
        # print (type(input_data))
        a=self.word_embedding.forward(input_data)*self.scale
        # print (type(a), cp.dtype(a))
        a=self.positional_encoding.forward(a)
        a=self.dropout.forward(a,training)
        
        # print (type(a), cp.dtype(a))
        for encoder_layer in self.encoder_layers:
            a=encoder_layer.forward(a, mask)
        # print (type(a), cp.dtype(a))
        return (a)
    
    
    def backward(self, error):
        global Iteration
        Iteration+=1
        for encoder_layer in reversed(self.encoder_layers):
            error=encoder_layer.backward(error)
            
        error=self.dropout.backward(error)
        error=self.positional_encoding.backward(error)*self.scale
        error=self.word_embedding.backward(error)
        
    def update_param(self):
        self.word_embedding.update_param()
        for encoder_layer in self.encoder_layers:
            encoder_layer.update_param()
        

def create_mask(dataQ, dataK ,decoder=True):
    
    batch_size, Qsize, Ksize= dataQ.shape[0], dataQ.shape[-1], dataK.shape[-1]

    # xb=cp.tril(xb, 0)
    if decoder:
        xm=cp.tril(cp.ones([batch_size , 1, Qsize , Ksize]), 0).astype(cp.int32)
    else:
        xm=cp.ones([batch_size , 1, Qsize , Ksize]).astype(cp.int32)

    dataQ=cp.array((dataQ!=0)*1).astype(cp.float32)
    
    dataK=cp.array((dataK!=0)*1).astype(cp.float32)
    
    
    # xm=xm*dataK.reshape([batch_size, 1, 1, -1])*dataQ.reshape([batch_size, 1, -1, 1])
    xm=xm*dataK.reshape([batch_size, 1, 1, -1])
    
    
    xm[xm==0]=-cp.inf
    xm=(xm-1).astype(cp.float32)

    # with np.printoptions(threshold=np.inf):
    #     # print (xb.shape)
    #     print (xm[:2,:,:15,:15])
    
    return xm
  
    
global Iteration
Iteration=0
    
# transformer.Model.config.counter=0

cfg=config()

# encoder=Encoder(transformer.Model.config)
# decoder=Decoder(transformer.Model.config)
# loss=Cross_Entropy(transformer.Model.config)

encoder=Encoder(cfg)
decoder=Decoder(cfg)
loss=Cross_Entropy(cfg)


X=transformer.Data.train.sentence_matrix
# print (X.shape)
cp.set_printoptions(suppress=True, precision=1, linewidth=1000)


for i in range (900):
    ta=time.time()
    # for j in range (500):
    for j in range (900): 
        
        ja=j
        
        # source=cp.array(X[0][ja]).astype(cp.int32)
        # target_a=cp.array(X[1][ja]).astype(cp.int32)
        
        source=X[0][ja]
        target_a=X[1][ja]
        
        # source_mask=create_mask(source,source, decoder=False)
        # target_mask=create_mask(target_a,target_a, decoder=True)
        # sideload_mask=create_mask(target_a,source, decoder=False)
        mask_a=[create_mask(source,source, decoder=False), create_mask(target_a,target_a, decoder=True),create_mask(target_a,source, decoder=False) ]
        
        
        # with np.printoptions(threshold=np.inf):
            # print (mask_a[2][:2,:,:15,:15])
    
        encoded_input=encoder.forward(source,mask_a[0] ,training=True,)
        
        # print (encoded_input.shape, target_a.shape)
        output_data, compacted_index=decoder.forward(encoded_input, target_a, mask_a[1: ] , training=True)
        
        # print (compacted_index.shape)
        """ shift the target by one to the left"""
        compacted_index=np.append(compacted_index[:,1:], cp.zeros(compacted_index.shape[0]).reshape([-1,1]), axis=1).astype(cp.int32)
        
        # print (compacted_index.shape, source.shape)
        
        loss_value, error, target=loss.forward(output_data, compacted_index)
        
        error_1, side_error =decoder.backward(error,target)
        encoder.backward(side_error)
        
        encoder.update_param()
        decoder.update_param()
        
        
        # ab=cp.argmax(output_data[0,5])
        
        # print (type(target))
        
        ac=cp.argmax(output_data, axis=2)
        ac=np.array(ac.get())
        
        target_a=np.array(target.reshape(ac.shape).get()).astype(np.int32)
        mask_a=(target_a!=0)*1
    
        mask_b=(ac==target_a)*mask_a
        # mask_b=(ac==target_a)*1
        
        # print (target_a, ac)
        
        # print (" ", "{:.2f}".format(loss_value), "",  "{:.2f}".format(np.sum(mask_b)/np.sum(mask_a)),"",j,"",(i*900)+j )
        print (" ", "{:.2f}".format(loss_value))
              
        # print (source[0], target_a[0])
        # print (loss_value, j, ab , output_data[0,5, int(ab)] , target_a[0,5])
        # # print (output_data.shape)
        # print ((time.time()-ta),loss_value)
        
    # print(time.time()-ta)
    
    
    
    
    
    
    # if j%3==2:
    #     path=i
    #     pickle_encoder = open(f'Weight/{path}encoder.pkl', 'wb')
    #     pickle_decoder = open(f'Weight/{path}decoder.pkl', 'wb')
    
    #     pkl.dump(encoder, pickle_encoder)
    #     pkl.dump(decoder, pickle_decoder)
    
    #     pickle_encoder.close()
    #     pickle_decoder.close()
        
    #     print(f'Saved to "{path}"')
    
    
    
# plt.plot(output_data[0,2,:].get())
# print (transformer.Model.config.counter)


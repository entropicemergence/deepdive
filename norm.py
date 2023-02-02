# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 06:10:00 2023

@author: gesit
"""

import cupy as cp






class Add_Norm:
    def __init__(self,config, dimension):
        self.batch_size=config.batch_size
        self.dtype=config.dtype
        self.epsilon=1e-6
        self.data=[]
        self.std=[]
        self.mean=[]
        self.jacobian=[]
        self.N=dimension
        self.Identity_Matrix_N=cp.expand_dims((cp.identity(self.N)*self.N)-1.0, axis=(0,1)).astype(self.dtype)
        self.error_front=[]
        
        # self.gamma=cp.ones([1,1,self.N]).astype(self.dtype)
        # self.beta=cp.zeros([1,1,self.N]).astype(self.dtype)
        
#         config.add_norm=True
        self.norm=[]
        self.error_front=[]
        
        
        self.DATA=[]
        self.STD=[]
        self.MEAN=[]
        
        self.N=cp.array([self.N]).astype(self.dtype)
        
        
        self.normalized_shape = config.opening_width
        self.normalized_axis = None
        self.build()
        
        
    def build(self):
        self.feature_size = None
        
        if self.normalized_shape is not None:
            self.gamma = cp.ones(self.normalized_shape).astype(self.dtype)
            self.beta = cp.zeros(self.normalized_shape).astype(self.dtype)

            self.vg, self.mg         = cp.zeros_like(self.gamma).astype(self.dtype), cp.zeros_like(self.gamma).astype(self.dtype)
            self.vg_hat, self.mg_hat = cp.zeros_like(self.gamma).astype(self.dtype), cp.zeros_like(self.gamma).astype(self.dtype)
            self.vb, self.mb         = cp.zeros_like(self.gamma).astype(self.dtype), cp.zeros_like(self.gamma).astype(self.dtype)
            self.vb_hat, self.mb_hat = cp.zeros_like(self.gamma).astype(self.dtype), cp.zeros_like(self.gamma).astype(self.dtype)
        
    def forward(self,input_data,skip_data):
        
#         self.DATA=input_data+skip_data
# #         self.STD=cp.sqrt((cp.std(self.DATA,2)**2)+self.epsilon).reshape([self.batch_size,-1,1])  #should be changed to more efficient
#         self.STD=(cp.std(self.DATA,2)+self.epsilon).reshape([self.batch_size,-1,1])  
#         self.MEAN=cp.mean(self.DATA,2).reshape([self.batch_size,-1,1])

#         self.norm=(self.DATA-self.MEAN)/self.STD
# #         print (self.norm.shape)
#         norm_a=(self.norm*self.gamma)+self.beta
        
        

        self.input_data = input_data+skip_data
        x_T = self.input_data.T
        
        if self.normalized_shape is None:
            self.normalized_shape = self.input_data.shape[1:]
            self.build()
        self.normalized_axis = tuple(cp.arange(self.input_data.ndim - self.gamma.ndim).tolist())
        
        self.feature_size = self.gamma.size
        
        self.mean = cp.mean(x_T, axis = 0)
        self.var = cp.var(x_T,axis = 0)
        
        self.X_centered = (x_T - self.mean)
        self.stddev_inv = 1 / cp.sqrt(self.var + self.epsilon)

        self.X_hat_T = self.X_centered * self.stddev_inv
        self.X_hat = self.X_hat_T.T
        
        self.output_data = self.gamma * self.X_hat + self.beta


        return self.output_data
        # return norm_a            #add gamma  and beta (dense witth initial 1 and 0)

    
    def backward(self,error_front):
# #         print(error_front.shape, self.norm.shape, self.gamma.shape)
#         self.error_front=error_front
#         error_f=self.error_front*self.gamma
        
#         error_f=cp.expand_dims(error_f,-2)
#         self.STD=cp.expand_dims(self.STD,-1)
#         self.DATA=cp.expand_dims(self.DATA,-1)
        
#         # print (cp.dtype(self.DATA),cp.dtype(self.STD))
#         error=(self.DATA-self.STD)
#         # print (error.shape,cp.dtype(error), cp.dtype(error_front),cp.dtype(self.STD))

#         """using  22.7 % memory, 35 s    using sub batch of 8    26.4 % and 14 s (max gpu utils)"""
#         for j in range (32//8):
#             # e=cp.squeeze(cp.matmul(error_f[j], (self.Identity_Matrix_N[0]-(cp.matmul(error[j],cp.moveaxis(error[j],-1,-2))/(cp.square(self.STD[j]))))/(self.N*self.STD[j])),1)
#             ja, jb= j*8, (j+1)*8
#             # if j==0:
#             #     e=cp.matmul(error_f[j], (self.Identity_Matrix_N[0]-(cp.matmul(error[j],cp.moveaxis(error[j],-1,-2))/(cp.square(self.STD[j]))))/(self.N*self.STD[j]))
#             #     e=cp.moveaxis(e,0,1)
#             # else:
#             #     e=cp.append(e, cp.moveaxis(cp.matmul(error_f[j], (self.Identity_Matrix_N[0]-(cp.matmul(error[j],cp.moveaxis(error[j],-1,-2))/(cp.square(self.STD[j]))))/(self.N*self.STD[j])),0,1),axis=0)
#             if j==0:
#                 e=cp.matmul(error_f[ja:jb], (self.Identity_Matrix_N-(cp.matmul(error[ja:jb],cp.moveaxis(error[ja:jb],-1,-2))/(cp.square(self.STD[ja:jb]))))/(self.N*self.STD[ja:jb]))
#             else:
#                 e=cp.append(e,cp.matmul(error_f[ja:jb], (self.Identity_Matrix_N-(cp.matmul(error[ja:jb],cp.moveaxis(error[ja:jb],-1,-2))/(cp.square(self.STD[ja:jb]))))/(self.N*self.STD[ja:jb])),axis=0)
#         e=cp.squeeze(e,2)
    
#         """using 43.7 % memory, 11.5 s"""
        
#         # print (type(self.N),type(error),type(self.STD),type(self.Identity_Matrix_N), type(error_f))
#         # print (cp.dtype(self.N),cp.dtype(error),cp.dtype(self.STD),cp.dtype(self.Identity_Matrix_N), cp.dtype(error_f))
#         # error=cp.squeeze(cp.matmul(error_f, (self.Identity_Matrix_N-(cp.matmul(error,cp.moveaxis(error,-1,-2))/(cp.square(self.STD))))/(self.N*self.STD)),2)

#         """using 57 % memeory, 11.5 s"""
#         # error=cp.matmul(error,cp.moveaxis(error,-1,-2))/(self.STD**2)
#         # error=(self.Identity_Matrix_N-error)/(self.N*self.STD)
#         # error=cp.squeeze(cp.matmul(error_f, error),2)
        
#         # print (cp.mean(error), cp.mean(e))
#         # error=error_front
        


        error_T = error_front.T
        #first variant
        output_error = (1 / self.feature_size) * cp.expand_dims(self.gamma, axis = self.normalized_axis).T * self.stddev_inv * (#self.gamma[np.newaxis, :].T
            self.feature_size * error_T
            - cp.sum(error_T, axis = 0)
            - self.X_centered * cp.power(self.stddev_inv, 2) * cp.sum(error_T * self.X_centered, axis = 0)
            )

        output_error = output_error.T
        
        self.grad_gamma = cp.sum(error_front * self.X_hat, axis = self.normalized_axis)
        self.grad_beta = cp.sum(error_front, axis = self.normalized_axis)
        
        return output_error
        # return e
        
    def update_param(self):    
        a=0
        
        # DW=cp.sum(cp.matmul(self.input_data.transpose(0, 2, 1), self.error_front), axis = 0)
        # DB=cp.sum(self.error_front, axis=(0,1))
        
        
        # self.m = self.beta*self.m + (1-self.beta)*DW
        # self.v = self.beta2*self.v+ (1-self.beta2)*(DW**2)
        # self.m_diff =self.m / (1-(self.beta**self.Iterationn))
        # self.v_diff=self.v / (1-(self.beta2**self.Iterationn))
        
        # a=((self.alpha*self.m_diff)/(cp.sqrt(self.v_diff)+self.epsilon))
        # self.gamma=self.gamma-a
        
        # self.m_b = self.beta*self.m_b + (1-self.beta)*DB
        # self.v_b = self.beta2*self.v_b+ (1-self.beta2)*(DB**2)
        # self.m_diff_b =self.m_b / (1-(self.beta**self.Iterationn))
        # self.v_diff_b=self.v_b / (1-(self.beta2**self.Iterationn))
        
        # b=((self.alpha*self.m_diff_b)/(cp.sqrt(self.v_diff_b)+self.epsilon))
        # self.beta=self.beta-b
        
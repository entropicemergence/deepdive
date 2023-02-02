# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 07:35:03 2023

@author: gesit
"""
import cupy as cp
from optimizer import Adam



class Softmax:
    def __init__(self, config):
        self.batch_size=config.batch_size
        self.softmax_out=[]
        self.unique_out_token=config.unique_out_token
        
        self.qk=[]
        
    def forward(self, qk):    #for multihead 2D softmax
    
        # qk_exp=cp.exp(qk)
        # self.qk=cp.divide(qk_exp, cp.sum(qk_exp,axis=(-2,-1)).reshape([qk.shape[0],-1,1,1]))
        
        # cp.set_printoptions(suppress=True, precision=2, linewidth=1000)
        
        qk_exp=cp.exp(qk-cp.max(qk, axis=-1, keepdims=True))
        
        # print (qk_exp[0,0,0::2,0::2])
        
        # self.qk=cp.divide(qk_exp, cp.sum(qk_exp,axis=(-1)).reshape([qk.shape[0],qk.shape[1],-1,1]))
        
        self.qk=cp.divide(qk_exp, cp.sum(qk_exp,axis=(-1)).reshape([qk.shape[0],qk.shape[1],-1,1]))
        
        
        # print (self.qk[0,0,0::2,0::2])
        
        # print (self.qk[0,0])
        
        return self.qk
    
    def softmax1D(self, input_data, target): #for final softmax
#         for j in range (32):               #changing the output value to match target for debugg
#             input_data[j,cp.arange(target.shape[1]),target[j].astype(cp.int32)]=5
#         input_data[0,5,500]=5
        
        input_data_exp=cp.exp(input_data)
        self.softmax_out=cp.divide(input_data_exp, cp.sum(input_data_exp,axis=(-1)).reshape([input_data_exp.shape[0],-1,1]))

        return self.softmax_out
    
    def softmax1D_back(self, error, target):  #target is index of training value, error is loss in softmax layer(equivalent to mean square error)
        n=target.shape[0]
        index=cp.arange(n).astype(cp.int32)
        self.softmax_out=self.softmax_out.reshape([n,-1])   # output fo softmax layer    


        Y_winner=self.softmax_out[index,target]

        e=-self.softmax_out * error.reshape([n,1]) * Y_winner.reshape([n,1])
        
        
        
        # cp.set_printoptions(suppress=True, precision=10, linewidth=1000)
        # print (cp.min(error,2)[:3])
        
        
        e[index,target]=Y_winner*(1-Y_winner)*error
        
        
        # print (error.shape, self.softmax_out.shape, target.shape,Y_winner.shape, ee.shape)
        

# #         plt.ylim(-0.01,0.01)
# #         plt.plot(self.softmax_out[5].get())
# #         plt.show()
        
#         a=(1.0/(error*self.unique_out_token)).reshape([-1,1])
#         error_2=-self.softmax_out*a
#         error_2[index, target]=(a*(1.0-a)).reshape([n])
#         error_2=error_2*error.reshape([-1,1])
        
# #         print (error.shape, target.shape, self.softmax_out.shape, error_2.shape, a.shape)
# #         print (error[:10],target[:10],self.softmax_out[0,:10])
# #         plt.ylim(-0.00002,0.00002)
# #         plt.plot(error_2[5].get())
#         return error_2.reshape([self.batch_size,-1,self.unique_out_token])
        
        # print (cp.max(e[0]))
        return e.reshape([self.batch_size,-1,self.unique_out_token])
    
    def backward(self,error):
        
#         n=error.shape[-1]       
#         self.qk=self.qk.reshape(self.batch_size,self.qk.shape[1],1,-1)
#         jacobian=(cp.identity(n).reshape(1,1,n,n)-self.qk)*cp.moveaxis(self.qk,-1,-2)
# #         print (error.shape, self.qk.shape, jacobian.shape,)        
#         e =cp.matmul(error.reshape(self.batch_size,-1,1,n), jacobian)


        # self.qk=self.qk.reshape(self.batch_size,self.qk.shape[1],-1)
        # dy_y=error*self.qk
        # e=((cp.sum(dy_y, axis=2).reshape(self.batch_size,-1,1))*-self.qk)+dy_y
        
        
        dy_y=error*self.qk
        e=((cp.sum(dy_y, axis=-1).reshape(self.batch_size, dy_y.shape[1] , -1 , 1 )) * -self.qk)+dy_y
        
        return e

'''

ensure any class variable has the right data type for compute

'''





class Dense:
    def __init__(self,config, input_shape, output_shape):
        self.batch_size=config.batch_size
        self.width=config.opening_width
        self.dtype=config.dtype
        
        std=1.0/cp.sqrt(input_shape)
        self.w=cp.random.uniform(-std,std, (input_shape,output_shape)).astype(self.dtype)
        
        # self.w=cp.random.normal(0,input_shape**-0.5, (input_shape,output_shape)).astype(self.dtype)
        
        
#         self.b=cp.random.normal(0,output_shape**-0.5, (output_shape)).astype(self.dtype)
        self.b=cp.zeros(output_shape).astype(self.dtype)
        
        self.input_data=[]
        self.error_front=[]
        
        self.DW=[]
        self.DB=[]
#         config.counter+=1

        # self.alpha = 0.01
        # self.alpha = 0.0011 
        
        # self.beta = 0.9
        # self.beta2 = 0.999 
        # self.epsilon = 0.000000001
        # self.Iterationn=1
        
        
        # self.m=cp.zeros([input_shape,output_shape]).astype(self.dtype)
        # self.m_diff=cp.zeros([input_shape,output_shape]).astype(self.dtype)
        # self.v=cp.zeros([input_shape,output_shape]).astype(self.dtype)
        # self.v_diff=cp.zeros([input_shape,output_shape]).astype(self.dtype)
        
        
        # self.m_b=cp.zeros(output_shape).astype(self.dtype)
        # self.m_diff_b=cp.zeros(output_shape).astype(self.dtype)
        # self.v_b=cp.zeros(output_shape).astype(self.dtype)
        # self.v_diff_b=cp.zeros(output_shape).astype(self.dtype)
        
        self.optimize_w=Adam(self.w)
        self.optimize_b=Adam(self.b)
        
        
        
    def forward(self, input_data):
        self.input_data=input_data
#         transformer.Model.config.counter+=1
        # print (self.w.shape, self.b.shape)
        return cp.dot(self.input_data,self.w)+self.b
    
    
    def backward(self,error):
        self.error_front=error
        
        # if self.error_front.shape[-1]==7802:
        #     # print (self.Iterationn)
        #     self.Iterationn+=1
        
        # print ("mean",cp.mean(error))
#         print (self.error_front.shape, self.input_data.shape, self.w.shape)

        self.DW=cp.sum(cp.matmul(self.input_data.transpose(0, 2, 1), self.error_front), axis = 0)
        self.DB=cp.sum(self.error_front, axis=(0,1))
        
        return cp.dot(self.error_front, self.w.T)


    def update_param(self):
        self.w=self.optimize_w.optimize(self.w, self.DW)
        self.b=self.optimize_b.optimize(self.b, self.DB)
        
        
#         print (Iteration)
    

        # e_std=cp.std(self.error_front)
        
        # if e_std > 0.5:
        # self.error_front=cp.clip(self.error_front, -0.2, 0.2)






        # print("{:.4f}".format(cp.std(self.error_front)), "", end='')
        # print("{:.3f}".format(cp.std(self.w)), " ", end='')
        
        
        
        
        
        # error_sum=(cp.sum(cp.abs(self.error_front)))*10000/self.error_front.size
        # print("{:.3f}".format(error_sum), "", end='')
        # print("{:.6f}".format(cp.std(self.w)), " ", end='') 
        
        # if error_sum < 0.1:
        #     c=0
        # else:
            
            

        
        # print (self.error_front.shape, self.w.shape, self.input_data.shape,DW.shape)
        
        
        
        # self.m = self.beta*self.m + (1-self.beta)*DW
        
        # self.v = self.beta2*self.v+ (1-self.beta2)*(DW**2)
        
        # self.m_diff =self.m / (1-(self.beta**self.Iterationn))
        
        # self.v_diff=self.v / (1-(self.beta2**self.Iterationn))
        
        # # print ("sum",cp.sum(self.w),  cp.sum(self.m), cp.sum(self.v), cp.sum(self.m_diff), cp.sum(self.v_diff), cp.sum(DW))
        
        # a=((self.alpha*self.m_diff)/(cp.sqrt(self.v_diff)+self.epsilon))
        # self.w=self.w-a
        
        
        # self.m_b = self.beta*self.m_b + (1-self.beta)*DB
        
        # self.v_b = self.beta2*self.v_b+ (1-self.beta2)*(DB**2)
        
        # self.m_diff_b =self.m_b / (1-(self.beta**self.Iterationn))
        
        # self.v_diff_b=self.v_b / (1-(self.beta2**self.Iterationn))
        
        # # print ("sum",cp.sum(self.w),  cp.sum(self.m), cp.sum(self.v), cp.sum(self.m_diff), cp.sum(self.v_diff), cp.sum(DW))
        
        # b=((self.alpha*self.m_diff_b)/(cp.sqrt(self.v_diff_b)+self.epsilon))
        # self.b=self.b-b
        
    
        
        
        # print("{:.1e}".format(cp.max(self.error_front)), "", end='')
        # print("{:.3f}".format(cp.max(self.error_front)), "", end='')
        # print("{:.3f}".format(cp.max(self.w)), " ", end='')

        # print (cp.sum(self.w), a.shape)
    


class Dropout:
    def __init__(self,config, dropout_rate=0.1):
        self.dtype=config.dtype
        self.dropout_rate=dropout_rate
        self.mask=[]
        
    def forward(self, input_data,training):
        self.mask=1.0
        if training:
            self.mask=cp.random.binomial(1,1-self.dropout_rate,input_data.shape).astype(self.dtype)
        return input_data*self.mask
    def backward(self, error):
#         training=True
#         print (self.mask.shape,error.shape)
        return error*self.mask
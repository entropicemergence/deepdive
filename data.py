# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 07:28:44 2023

@author: gesit
"""

import numpy as np
try:
    import cupy as cp
except:
    pass
import os
import ebooklib
import re
from ebooklib import epub
from bs4 import BeautifulSoup
from compress_pickle import dump, load
from matplotlib import image, pyplot as plt
from PIL import Image
from tqdm import tqdm

# import pickle




from tokenizers import Tokenizer, pre_tokenizers, normalizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# from tokenizers.processors import TemplateProcessing




# class CfgNode:
class CN:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CN):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)
    def to_dict(self):
        return { k: v.to_dict() if isinstance(v, CN) else v for k, v in self.__dict__.items() }
    def merge_from_dict(self, d):
        self.__dict__.update(d)



class text_process:
    def __init__(self, data_folder):
        self.banned_char='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        self.additional_character={'<beg>': [1,0], '<end>': [2,0], '<pad>': [0,0], '<mystery>': [3,0]}
        self.data_kinds=["test","train","val"]
        self.data_language=["en","de"]
        self.data_folder=data_folder
        self.batch_size=32
        
    def filter_banned(self, word):
        new_word=""
        for char in word:
            if char not in self.banned_char:
                new_word=new_word+char
        return (new_word.lower()) 
    
    def build_vocab(self, list_data):
    
        list_data=self.clean_text(list_data)
        vocab={}
        for j in (list_data):
            for jj in (j):
                if jj not in (vocab):
                    vocab[jj]=1
                else:
                    vocab[jj]+=1
        
        vocab_a=sorted(vocab.items(),key= lambda x:x[1],reverse=True)
        for ja, j in enumerate (vocab_a,0):
            if j[1]==1:
                break
        vocab_b=vocab_a[:ja]
        vocab_c=self.additional_character.copy()    
        for n, voc_a in enumerate (vocab_b,4):
            vocab_c[voc_a[0]]=[n,voc_a[1]]       
        return (vocab_c)
    
    def clean_text(self,text):
        for a1x, a1 in enumerate (text,0):
            for a2x, a2 in enumerate (a1,0): 
                text[a1x][a2x]=self.filter_banned(a2)
        return text
    
    
    def build_sentence(self,text):
#         n=2
        text=self.clean_text(text)
#         print (text[n])
        return (text)

    def matrix_form(self,text,enTrue=True):
        max_a=max(len(l) for l in text)
        textb=np.zeros([len(text),max_a+2])
        if enTrue:
            en=0
        else:
            en=1
        
        for i, sentence in enumerate (text,0):
            textb[i][0]=1
            for j, word in enumerate (sentence,1):
                try:
                    textb[i][j]=self.Data.train.vocabulary[en][word][0]
                except:
                    textb[i][j]=3
            textb[i][j+1]=2
        l=textb.shape[0]//self.batch_size
        return textb[:l*self.batch_size].reshape(l,self.batch_size,textb.shape[1])
    
    def prepare_data(self):
        self.Data=CN()
        self.Data.val=CN()
        self.Data.test=CN()
        self.Data.train=CN()

        en=[data.split() for data in open("numpy-transformer-master/dataset/val.en", 'r', encoding='utf-8')]
        de=[data.split() for data in open("numpy-transformer-master/dataset/val.de", 'r', encoding='utf-8')]
        self.Data.val.vocabulary=[self.build_vocab(en),self.build_vocab(de)]
        self.Data.val.sentence=[self.build_sentence(en),self.build_sentence(de)]
        self.Data.val.sentence_matrix=[self.matrix_form(self.Data.val.sentence[0],enTrue=True),
                                       self.matrix_form(self.Data.val.sentence[1],enTrue=False)]
        
        en=[data.split() for data in open("numpy-transformer-master/dataset/test.en", 'r', encoding='utf-8')]
        de=[data.split() for data in open("numpy-transformer-master/dataset/test.de", 'r', encoding='utf-8')]
        self.Data.test.vocabulary=[self.build_vocab(en),self.build_vocab(de)]
        self.Data.test.sentence=[self.build_sentence(en),self.build_sentence(de)]
        self.Data.test.sentence_matrix=[self.matrix_form(self.Data.test.sentence[0],enTrue=True),
                                        self.matrix_form(self.Data.test.sentence[1],enTrue=False)]
        
        en=[data.split() for data in open("numpy-transformer-master/dataset/train.en", 'r', encoding='utf-8')]
        de=[data.split() for data in open("numpy-transformer-master/dataset/train.de", 'r', encoding='utf-8')]
        self.Data.train.vocabulary=[self.build_vocab(en),self.build_vocab(de)]
        self.Data.train.sentence=[self.build_sentence(en),self.build_sentence(de)]
        
        self.Data.train.sentence_matrix=[self.matrix_form(self.Data.train.sentence[0],enTrue=True),
                                         self.matrix_form(self.Data.train.sentence[1],enTrue=False)]

        return (self.Data)
    
    

class file_process:
    def __init__(self):
        self.a=0
        
    def list_files(self, path):
        # List all files and subdirectories in the specified directory
        # pathh = ['D:\Book\Manga\Snafu']
        # print (path)
        pathh = path
        # path='.'
        file_list=[]
        dir_list=[]
        
        
        for path in pathh:
            # print (path)
            for item in os.listdir(path):
                # Check if item is a file or directory
            #     print (os.path.isfile(path+"/"+item))    
                if os.path.isfile(path+"/"+item):
                    file_list.append(path+"/"+item)
            #         print(f"{item} is a file")
                elif os.path.isdir(path+"/"+item):
                    c=0
                    pathh.append(path+"/"+item)
            #         print(f"{item} is a directory")
        return pathh, file_list
    
    def read_epub(self, path):
        book = epub.read_epub(path)
        t=""
        # Iterate over all the resources in the book
        for item in book.get_items():
          # Check if the resource is a chapter
          if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Extract the text from the chapter
            text = item.get_content()
            # Decode the text from bytes to a string
            text = text.decode('utf-8')
            # Use BeautifulSoup to parse the HTML and extract the readable text
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
            # Use a regular expression to remove unreadable symbols
            text = re.sub(r'[^\x00-\x7F]+',' ', text)
            t+=text
        return (t)
            
    def tokenize(self, file_name, vocab_s=4096):
        
        tokenizer=Tokenizer(BPE())
        tokenizer.normalizer=normalizers.Sequence([normalizers.Lowercase(), normalizers.NFD(), normalizers.StripAccents()])
        tokenizer.pre_tokenizer= pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True), 
                                                          pre_tokenizers.ByteLevel(add_prefix_space=True)])  #pre_tokenizers add split word identifier
        
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True) #By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don’t want the offsets to include these
        
        train=BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size = vocab_s, min_frequency=2)
        
        tokenizer.train([file_name], train)
        
        # tokenizer.
        # out=tokenizer.encode("sdsd")

        return tokenizer
        
    def write_text(self, file_name, data):
        with open(file_name, 'w') as outfile:
            outfile.write(data)
    def read_text(self, file_name):
        ff=""
        with open(file_name, 'r') as infile:
            for f in infile.readlines():
                ff+=f
        return ff
    def save_gzip(self, data, file_name):
        dump(data, file_name+".gz")
    def load_gzip(self, file_name):
        return load(file_name+".gz")
    
class analysis:
    def __init__(self):
        pass
    # def multi_implot(self, data, shape):
        
    #     fig,ax=plt.subplots(1,6,figsize=(18,6))
    #     for kk in range(6):
    #         ax[kk].imshow(data_a[kk])
            
class mask:
    def __init__(self):
        pass
    def diagonal(self, n):
        b=cp.triu(np.ones([n,n]),1).reshape([1,1,n,n])
        b[b==1]=-cp.inf
        return b
    # def create_mask(dataQ, dataK ,decoder=True):
        
    #     batch_size, Qsize, Ksize= dataQ.shape[0], dataQ.shape[-1], dataK.shape[-1]

    #     # xb=cp.tril(xb, 0)
    #     if decoder:
    #         xm=cp.tril(cp.ones([batch_size , 1, Qsize , Ksize]), 0).astype(cp.int32)
    #     else:
    #         xm=cp.ones([batch_size , 1, Qsize , Ksize]).astype(cp.int32)

    #     dataQ=cp.array((dataQ!=0)*1).astype(cp.float32)
        
    #     dataK=cp.array((dataK!=0)*1).astype(cp.float32)
        
    #     # xm=xm*dataK.reshape([batch_size, 1, 1, -1])*dataQ.reshape([batch_size, 1, -1, 1])
    #     xm=xm*dataK.reshape([batch_size, 1, 1, -1])

    #     xm[xm==0]=-cp.inf
    #     xm=(xm-1).astype(cp.float32)

    #     # with np.printoptions(threshold=np.inf):
    #     #     # print (xb.shape)
    #     #     print (xm[:2,:,:15,:15])
        
    #     return xm
    
    # mask_a=[create_mask(source,source, decoder=False), create_mask(target_a,target_a, decoder=True),create_mask(target_a,source, decoder=False) ]



class Images:
    def __init__(self):
        pass
    def load_image(self, path):
        return image.imread(path)
        # I=Image.open(path)
        # I.crop
        
        # return 
    def show_image(self, img, size=4):
        plt.figure(figsize=(size,size))
        plt.imshow(img)
        plt.show()
        
    def show_image_batch(self,img, per_row):
        # width=18
        batch_num=img.shape[0]
        a=(batch_num//per_row)
        n=0
        for aa in range(a):
            fig,ax=plt.subplots(1,per_row,figsize=(18,18))
            for kk in range(per_row):
                ax[kk].imshow(img[n])
                n+=1
        
    def crop_top(self,img):
        return (img[:img.shape[1],:])
    # def reshape(self,img, shape )
    def files_to_array(self, folder, shape):
        '''output square crop colored image, not including black/white'''
        F=file_process()
        
        _, fl=F.list_files([folder])

        target_shape=shape
        image_dataset=np.empty([0,target_shape,target_shape,3]).astype(np.uint8)
        fl=tqdm(fl)
        
        for fa in fl:
            try:
        #         print (fa)
                img=self.load_image(fa)
        #         print (img.shape)
        
                x=img.shape[1]
                y=img.shape[0]
                if y>x:
                    img=img[:x,:,:3]
                else:
                    img=img[:,:y,:3]
        
                if fa[-3:]=="png":
            #         print ("png")
                    img=(img*255).astype(np.uint8)
                else:
                    pass
                img=Image.fromarray(img)
                img.thumbnail((target_shape,target_shape))
                img=np.array(img)[np.newaxis,...].astype(np.uint8)
        #         print (img.shape)
                image_dataset=np.append(image_dataset, img, axis=0)
                       
            except Exception as e: 
                print(e)
        return image_dataset
        
        # plt.figure(figsize=(10,10))
        # I.show_image(img)
                
        
        
    
    
    
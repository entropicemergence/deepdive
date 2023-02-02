# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 20:15:52 2023

@author: gesit
"""

# import matplotlib.pyplot as plt
# import time
import cupy as cp
import numpy as np
# import pickle as pkl
from tqdm import tqdm

from data import CN, text_process, file_process, mask

# from layer import Softmax,  Dense, Dropout
# from encoding import positional_encoding, word_embedding
# from transformer import Pure_Decoder_Layer
from loss import Cross_Entropy
from LM import LM

cp.set_printoptions(suppress=True, precision=1, linewidth=1000)





def config():
    C=CN()
    C.batch_size=8
    C.max_input_length=128
    C.opening_width=256
    C.decoder_layer=3
    C.decoder_heads=8
    
    C.head_width=32
    C.feed_forward_width=512
    
    C.unique_in_token=4096
    C.unique_out_token=4096
    C.dtype=cp.float32
    C.dttype2=cp.int32
    return C



C=config()
F=file_process()
M=mask()

T=F.load_gzip("tokenizer")
encoded_text=F.load_gzip("encoded_train_data")

lm=LM(C)
loss=Cross_Entropy(C)

seq_length=C.max_input_length

mask=M.diagonal(C.max_input_length)

k=2846689//((seq_length+1)*C.batch_size)

qq=np.arange(k-1)
J=tqdm(qq)

f=file_process()





# -*- coding: utf-8 -*-
"""
Demonstrates very basic use of ImageItem to display image data inside a ViewBox.
"""

## Add path to library (just for examples; you do not need this)
# import initExample

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime

app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: ImageItem')
view = win.addViewBox()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(border='w')
view.addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 8*128, 128))

## Create random image
# data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)

# data=np.random.normal(size=(15, 128*8, 128), loc=1024, scale=64).astype(np.uint16)

data=[]
i = 0

updateTime = ptime.time()
fps = 0



j=0
def updateData():
    global img, data, i, updateTime, fps, j

    ## Display the data
    # img.setImage(data[i])
    
    
    # i = (i+1) % data.shape[0]


    start=(j*(seq_length+1))*C.batch_size
    end=start+((seq_length+1)*C.batch_size)
    text_a=np.array(encoded_text.ids[start:end]).reshape([C.batch_size,-1])
    output_data, _ =  lm.forward(text_a[:,0:-1], mask, training=True)
    loss_value, error, target = loss.forward(output_data, text_a[:,1:])
    lm.backward(error,target)  
    lm.update_param()
    
    
    a=cp.ones([128*8,10]).astype(cp.float32)
    a=cp.append(a, lm.decoder_layers[0].masked_multihead_attention.qk[0].reshape([128*8,128]), axis=1 )
    a=cp.append(a, lm.decoder_layers[1].masked_multihead_attention.qk[0].reshape([128*8,128]), axis=1 )
    a=cp.append(a, lm.decoder_layers[2].masked_multihead_attention.qk[0].reshape([128*8,128]), axis=1 )
    
    
    data=np.array(a.get())

    # data=lm.decoder_layers[0].masked_multihead_attention.qk[0,2]
    img.setImage(data)
    
    
    if j==2500:
        j=0
    j+=1



    QtCore.QTimer.singleShot(1, updateData)
    now = ptime.time()
    fps2 = 1.0 / (now-updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1
    
    #print "%0.1f fps" % fps
    

updateData()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()




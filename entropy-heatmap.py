from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


aa = net.blobs['score'].data[0]

def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
aasfmx = softmax(aa)

lst = list(range(19))

entropy=0;

for i in lst:
    entropy = entropy+(aasfmx[i,:,:] * np.log2(aasfmx[i,:,:]))
    
entropy = -1 * entropy; #entropy is -sum( pi * log pi)
    
plt.imsave('image.png', entropy, cmap=plt.cm.jet)

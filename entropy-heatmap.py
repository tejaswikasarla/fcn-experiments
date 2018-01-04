from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

aa_out = net.blobs['score'].data[0] #last layer of fcn output
aa_sfmx = softmax(aa_out)
lst = list(range(19)) #no of classes of cityscapes dataset

#find entropy of the image (of its prediction of all 19 classes from softmax
# entropy is -sum_i(p_i *log p_i))
entropy=0;
for i in lst:
    entropy = entropy+(aa_sfmx[i,:,:] * np.log2(aa_sfmx[i,:,:]))
entropy = -1 * entropy; #entropy is -sum( pi * log pi)

#save without colorbar reference
#plt.imsave('image.png', entropy, cmap=plt.cm.jet) 

#save with colorbar reference
fig,ax=plt.subplots(figsize=(20.48,10.24))
im=ax.imshow(entropy,cmap=plt.cm.jet)
ax.axis('off')
fig.colorbar(im,shrink=0.75)
fig.savefig('images.png',bbox_inches='tight',pad_inches=0.0)

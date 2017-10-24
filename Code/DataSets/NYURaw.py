import os.path
import random
import numpy as np
import numpy.linalg as LA
from PIL import Image

path = '../Code/DataSets/NYURaw/'

class Dataset:
    
    def __init__(self,subset=1.0):    
        self.Indices = []
        for i in range(1,252885):
            rgb = path+'RGB/'+str(i).zfill(6)+'.png'
            norm = path+'NORM/'+str(i).zfill(6)+'.png'
            if os.path.isfile(rgb) and os.path.isfile(norm):
                self.Indices.append(i)
            #else:
            # print('Frame No. ',str(i),' IS MISSING')
        random.seed(7)
        random.shuffle(self.Indices)
        self.size = int(subset*len(self.Indices))
    
    def get_batch(self,batch_size=32):
        # Initialisation
        images = np.zeros([batch_size, 240, 320, 3], dtype=np.float32)
        normals = np.zeros([batch_size, 240, 320, 3], dtype=np.float32)
        # Loading images
        for i in range(batch_size):
            if len(self.Indices): 
                ni = self.Indices.pop()
                images[i,:,:,:] = np.asarray(Image.open(path+'RGB/'+str(ni).zfill(6)+'.png'))
                normals[i,:,:,:] = np.asarray(Image.open(path+'NORM/'+str(ni).zfill(6)+'.png'))
        
        # Convert to range [-1,+1]
        normals = ((normals / 254)*2)-1
        # Mask of valid normals
        norml2 = LA.norm(normals,axis=3)
        mask = (norml2 > 0.5).astype(np.float32)
        mask3 = np.repeat(np.expand_dims(mask,3),3,3)
        # Normalise
        normals = np.divide(normals,np.expand_dims(norml2,3))
        normals = np.nan_to_num(normals)
        # Apply mask
        normals = np.multiply(normals,mask3)
        
        return images,normals
import h5py
import numpy as np
from random import shuffle

class Dataset:
    def __init__(self, file):
        
        self.dataset = h5py.File(file,'r')
        
        # (index, height, width, channels)
        self.images = np.array(dataset['Images']).transpose([3,2,1,0])
        self.normals = np.array(dataset['Normals']).transpose([3,2,1,0])
        self.masks = np.array(dataset['Masks']).transpose([2,1,0])
        self.rotations = np.array(dataset['Rotations']).transpose([2,0,1])
        # Image height and width
        self.width = self.images.shape[2]
        self.height = self.images.shape[1]
        # Number of samples in the dataset 
        self.size = images.shape[0]
        # Queue for choosing the samples
        self.queue = []
        
    def _random_crop(self, index, width, height):
        
        maxHeightIndex = self.height - height
        heightIndex = np.random.randint(0,maxHeightIndex)
    
        maxWidthIndex = self.width - width
        widthIndex = np.random.randint(0,maxWidthIndex)
        
        imgCrop = images[index, heightIndex:heightIndex+height, widthIndex:widthIndex+width, :]
        normCrop = normals[index, heightIndex:heightIndex+height, widthIndex:widthIndex+width, :]
        maskCrop = masks[index, heightIndex:heightIndex+height, widthIndex:widthIndex+width]
        
        return imgCrop, normCrop, maskCrop
    
    def _next_index(self):
        
        if (len(self.queue) == 0):
            self.queue = list(range(self.size))
            shuffle(self.queue)
        
        return self.queue.pop()
        
    def get_batch(self, batch_size, width, height):
        
        images = np.empty([batch_size, height, width, 3], dtype=np.float32)
        normals = np.empty([batch_size, height, width, 3], dtype=np.float32)
        masks = np.empty([batch_size, height, width, 1], dtype=np.uint8)
        
        for i in range(batch_size):
            ni = self._next_index()
            images[i,:,:,:], normals[i,:,:,:], masks[i,:,:,0] = self._random_crop(ni, width, height)
        
        return images, normals, masks
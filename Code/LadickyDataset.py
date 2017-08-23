import h5py
import numpy as np
import tensorflow as tf
from random import shuffle

class LadickyDataset:
    def __init__(self, file, subset=None,
                 batch_res=(240,320), scale=(1.0,1.5), flip=True, rotation=5, color=True, brightness=True):
        '''
        subset: A list of the index of images
        '''
        
        self.dataset = h5py.File(file,'r')
        
        # Load images and normals (index, height, width, channels)
        self.images = np.array(self.dataset['images']).transpose([0,3,2,1])
        self.normals = np.array(self.dataset['normals']).transpose([0,3,2,1])

        # Dataset height and width
        self.width = self.images.shape[2]
        self.height = self.images.shape[1]
        # Output height and width
        self.batch_height = batch_res[0]
        self.batch_width = batch_res[1]
        
        #Scaling factors in respect to batch resolution
        if(min(scale) < 1.0):
            print("Error: Scaling factor is in respect to batch_res and must be greater than 1.0 ")
        else:
            self.scale = scale
        
        # Subset 
        if subset == None: # Use all images
            self.validIndices = list(range(self.images.shape[0]))
        else: # Use a subset of dataset 
            self.validIndices = subset
        
        # Number of usable samples in the dataset 
        self.size = len(self.validIndices)
        
        # Random flip
        self.flip = flip
        
        # Random color changes
        self.change_color = color
        self.maxHueDelta = 0.1
        self.maxSatDelta = 0.5
        
        # Queue for choosing the samples
        self.queue = []
        
        # Building the computional graph
        self._build_tf_graph()
    
    def _build_tf_graph(self):
        
        # Creating the tf session
        self.sess = tf.Session()
        
        # Input placeholders
        self.tfImgs = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 3))
        self.tfNorms = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 3))
                
        # Scaling
        # Randomly chooses a scaling factor
        scales = tf.convert_to_tensor(self.scale)
        rand_index = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
        rand_scale = scales[rand_index]
        # Scales
        size = tf.cast([self.batch_height*rand_scale, self.batch_width*rand_scale], tf.int32)
        reszImgs = tf.image.resize_images(self.tfImgs, size)
        reszNorms = tf.image.resize_images(self.tfNorms, size)
        normNorms = tf.nn.l2_normalize(reszNorms,3)
        
        # Random Crop
        # Random height offset 
        maxHeightIndex = size[0] - 240
        heightIndex = tf.random_uniform([], minval=0, maxval=maxHeightIndex+1, dtype=tf.int32)
        # Random width offset
        maxWidthIndex = size[1] - 320
        widthIndex = tf.random_uniform([], minval=0, maxval=maxWidthIndex+1, dtype=tf.int32)
        # Crops
        cropImgs = tf.image.crop_to_bounding_box(reszImgs, heightIndex, widthIndex, 240, 320)
        cropNorms = tf.image.crop_to_bounding_box(normNorms, heightIndex, widthIndex, 240, 320)
        
        # Flip , this is a lazy definition, its excution depends on the rand_flip
        flipImgs = tf.reverse(cropImgs,[2])
        revNorms = tf.reverse(cropNorms,[2])
        flipNorms = tf.multiply([-1.0,1.0,1.0],revNorms)
        # Random flip
        rand_flip = tf.cast(tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32), tf.bool)
        randFlipImgs = tf.cond(rand_flip, lambda: flipImgs, lambda: cropImgs) #Flip or last value
        randFlipNorms = tf.cond(rand_flip, lambda: flipNorms, lambda: cropNorms) # Flip or last value
        
        # Random color changes
        change_color = tf.cast(self.change_color, tf.bool)
        # Delta values
        hueDelta = tf.random_uniform([], -self.maxHueDelta, self.maxHueDelta)
        satFactor = tf.random_uniform([], 1.0-self.maxSatDelta, 1.0+self.maxSatDelta)     
        # Convert image RGB values to [0,1] range
        rngImgs = tf.clip_by_value(tf.divide(randFlipImgs, 255.0), 0.0, 1.0) 
        # Convert RGB images to HSV 
        hsvImgs = tf.image.rgb_to_hsv(rngImgs)
        hue = tf.slice(hsvImgs, [0, 0, 0, 0], [-1, -1, -1, 1])
        saturation = tf.slice(hsvImgs, [0, 0, 0, 1], [-1, -1, -1, 1])
        value = tf.slice(hsvImgs, [0, 0, 0, 2], [-1, -1, -1, 1])
        # Change hue and saturation
        hue = tf.cond(change_color, lambda: tf.mod(hue + (hueDelta + 1.), 1.), lambda: hue)
        saturation = tf.cond(change_color, lambda: tf.clip_by_value(saturation*satFactor, 0.0, 1.0), lambda: saturation)
        # Convert to RGB
        hsv = tf.concat([hue, saturation, value], 3)
        colorImgs = tf.image.hsv_to_rgb(hsv)
        
        # Outputs
        self.tfOutImgs = tf.image.convert_image_dtype(colorImgs, tf.uint8, saturate=True)
        self.tfOutNorms = randFlipNorms
    
    def _next_index(self):
        
        if (len(self.queue) == 0):
            self.queue = self.validIndices[:]
            shuffle(self.queue)
        
        return self.queue.pop()
        
    def get_batch(self, batch_size=32):
        ''' A batch with data augmentation '''
        images = np.empty([batch_size, self.height, self.width, 3], dtype=np.float32)
        normals = np.empty([batch_size, self.height, self.width, 3], dtype=np.float32)
        
        for i in range(batch_size):
            ni = self._next_index()
            images[i,:,:,:] = self.images[ni,:,:,:]
            normals[i,:,:,:] = self.normals[ni,:,:,:]
        
        (outImgs, outNorms) = self.sess.run((self.tfOutImgs,self.tfOutNorms),
                                            feed_dict={self.tfImgs: images, self.tfNorms: normals})
        return outImgs, outNorms
    
    def get_data(self, index):
        ''' Resized original data: Image and normal map'''
        
        image = self.images[index,:,:,:]
        normals = self.normals[index,:,:,:]

        tfSize = tf.constant([self.batch_height,self.batch_width], dtype=tf.int32)
        tfImg = tf.constant(image)
        tfNorms = tf.constant(normals)         
        reszImgs = tf.image.resize_images(tfImg, tfSize)
        reszNorms = tf.image.resize_images(tfNorms, tfSize)
        normNorms = tf.nn.l2_normalize(reszNorms, 2)
        (outImg, outNorms) = self.sess.run((reszImgs, normNorms))
                
        return outImg, outNorms
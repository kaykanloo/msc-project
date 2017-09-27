import h5py
import numpy as np
import tensorflow as tf
import random

file = 'DataSets/MAT/NYUAltDataSet.mat'
trainNdxs = [2,3,4,5,6,7,9,10,11,12,18,19,21,22,23,24,25,26,43,44,47,48,49,50,51,52,53,54,57,63,64,65,66,67,68,69,70,71,72,73,74,79,80,81,82,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,119,120,121,122,123,129,134,135,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,155,156,157,158,159,160,161,162,163,164,165,169,176,177,178,202,203,204,205,212,213,214,215,216,217,218,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,250,251,252,253,254,255,256,257,258,259,260,261,262,264,265,266,267,268,269,273,274,275,276,277,285,286,287,288,289,290,291,292,293,294,302,303,304,305,306,307,308,312,313,317,318,319,320,321,322,323,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,352,353,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,390,391,392,393,397,398,399,400,401,402,403,404,405,406,407,408,409,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,435,436,437,438,439,448,449,450,451,452,453,454,455,456,457,458,459,460,466,467,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,513,526,527,528,529,533,534,535,539,540,541,542,543,544,545,546,547,551,552,553,571,572,573,574,575,576,577,583,584,585,586,587,588,589,594,595,596,597,598,599,600,601,607,608,609,610,613,614,615,621,622,623,624,625,626,627,628,629,630,631,638,639,640,641,642,645,646,647,648,651,652,653,654,658,659,660,661,664,665,666,673,674,681,682,683,684,690,691,694,695,699,700,701,702,703,704,713,714,715,718,719,720,721,722,728,729,734,735,736,737,738,739,740,741,744,745,746,747,748,749,750,751,752,753,754,755,756,757,787,788,789,790,791,792,793,794,795,796,797,798,804,805,806,807,808,814,815,816,817,818,819,823,824,825,826,827,828,829,830,831,846,847,848,852,853,854,855,862,863,864,865,866,867,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,908,909,910,911,912,913,914,915,919,920,921,922,923,924,928,929,930,935,936,937,938,939,940,941,942,943,947,948,949,950,951,952,953,954,955,956,957,962,963,967,968,977,978,979,980,981,982,983,984,985,986,987,988,989,995,996,997,998,999,1004,1005,1006,1007,1008,1012,1013,1014,1015,1016,1017,1018,1019,1023,1024,1025,1026,1027,1028,1029,1030,1034,1035,1036,1039,1040,1041,1042,1043,1044,1045,1046,1049,1050,1053,1054,1055,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1084,1085,1086,1096,1104,1109,1110,1111,1112,1113,1114,1115,1119,1120,1121,1131,1132,1133,1136,1137,1138,1139,1140,1141,1142,1158,1159,1160,1167,1168,1171,1172,1176,1177,1184,1185,1186,1187,1188,1189,1190,1196,1197,1198,1199,1212,1213,1214,1220,1221,1222,1223,1224,1230,1231,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1250,1251,1252,1265,1266,1267,1268,1269,1270,1271,1272,1273,1280,1281,1282,1283,1295,1299,1300,1308,1309,1310,1311,1312,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1332,1333,1340,1341,1342,1343,1344,1345,1349,1350,1351,1356,1357,1358,1359,1360,1361,1362,1365,1366,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1391,1392,1401,1402,1403,1404,1405,1414,1415,1416,1417,1418,1419,1424,1425,1426,1427,1428,1433,1434,1435,1436,1437,1438,1439]


class Dataset:
    def __init__(self, subset= trainNdxs, seed=1,
                 batch_res=(240,320), scale=(1.0,1.5), flip=True, color=True):
        '''
        subset: A list of the index of images
        '''
        # Load dataset file
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

        # Random seed
        self.seed = seed
        random.seed(seed)

        # Building the computional graph
        self._build_tf_graph()

    def _build_tf_graph(self):

        # Creating the tf session
        tf.set_random_seed(self.seed)
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
            random.shuffle(self.queue)

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

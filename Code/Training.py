# Import Data Set
import sys 
import os
sys.path.append(os.path.join(os.getcwd(), '../Code/'))
from LadickyDataset import *
# Import Tensorflow and Keras
import tensorflow as tf
from keras.models import  Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input , Flatten, Dense, Reshape, Lambda
from keras.layers.convolutional import Conv2D
# Import other modules 
from math import ceil
from PIL import Image
import argparse
import time

# Utility functions
def show_image(npimg):
    return Image.fromarray(npimg.astype(np.uint8))
def show_normals(npnorms):
    return Image.fromarray(((npnorms+1)/2*255).astype(np.uint8))

# Loss function
def mean_dot_product(y_true, y_pred):
    dot = tf.einsum('ijkl,ijkl->ijk', y_true, y_pred) # Dot product
    n = tf.cast(tf.count_nonzero(dot),tf.float32)
    mean = tf.reduce_sum(dot) / n
    return -1 * mean

# Model definition
def vgg16_model():
    # create model
    input_tensor = Input(shape=(240, 320, 3)) 
    base_model = VGG16(input_tensor=input_tensor,weights='imagenet', include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(80*60*3, activation='relu', name='fc2')(x)
    x = Reshape((60,80,3))(x)
    x = Lambda(lambda x: tf.image.resize_bilinear(x , [240,320]) )(x)
    pred = Lambda(lambda x: tf.nn.l2_normalize(x, 3) )(x)
    model = Model(inputs=base_model.input, outputs=pred)
    # Compile model
    model.compile(loss= mean_dot_product, optimizer='sgd')
    return model

# Training
if __name__ == "__main__":
    
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("ExperimentID", help="A name for current experiment. Also, it will be used as a prefix for output file names.")
    args = parser.parse_args()
    
    # Load data set
    file = '../Data/LadickyDataset.mat'
    trainNdxs = [3,4,5,6,7,8,10,11,12,13,19,20,22,23,24,25,26,27,44,45,48,49,50,51,52,53,54,55,58,64,65,66,67,68,69,70,71,72,73,74,75,80,81,82,83,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,120,121,122,123,124,130,135,136,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,156,157,158,159,160,161,162,163,164,165,166,170,177,178,179,203,204,205,206,213,214,215,216,217,218,219,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,251,252,253,254,255,256,257,258,259,260,261,262,263,265,266,267,268,269,270,274,275,276,277,278,286,287,288,289,290,291,292,293,294,295,303,304,305,306,307,308,309,313,314,318,319,320,321,322,323,324,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,353,354,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,391,392,393,394,398,399,400,401,402,403,404,405,406,407,408,409,410,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,436,437,438,439,440,449,450,451,452,453,454,455,456,457,458,459,460,461,467,468,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,514,527,528,529,530,534,535,536,540,541,542,543,544,545,546,547,548,552,553,554,572,573,574,575,576,577,578,584,585,586,587,588,589,590,595,596,597,598,599,600,601,602,608,609,610,611,614,615,616,622,623,624,625,626,627,628,629,630,631,632,639,640,641,642,643,646,647,648,649,652,653,654,655,659,660,661,662,665,666,667,674,675,682,683,684,685,691,692,695,696,700,701,702,703,704,705,714,715,716,719,720,721,722,723,729,730,735,736,737,738,739,740,741,742,745,746,747,748,749,750,751,752,753,754,755,756,757,758,788,789,790,791,792,793,794,795,796,797,798,799,805,806,807,808,809,815,816,817,818,819,820,824,825,826,827,828,829,830,831,832,847,848,849,853,854,855,856,863,864,865,866,867,868,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,909,910,911,912,913,914,915,916,920,921,922,923,924,925,929,930,931,936,937,938,939,940,941,942,943,944,948,949,950,951,952,953,954,955,956,957,958,963,964,968,969,978,979,980,981,982,983,984,985,986,987,988,989,990,996,997,998,999,1000,1005,1006,1007,1008,1009,1013,1014,1015,1016,1017,1018,1019,1020,1024,1025,1026,1027,1028,1029,1030,1031,1035,1036,1037,1040,1041,1042,1043,1044,1045,1046,1047,1050,1051,1054,1055,1056,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1085,1086,1087,1097,1105,1110,1111,1112,1113,1114,1115,1116,1120,1121,1122,1132,1133,1134,1137,1138,1139,1140,1141,1142,1143,1159,1160,1161,1168,1169,1172,1173,1177,1178,1185,1186,1187,1188,1189,1190,1191,1197,1198,1199,1200,1213,1214,1215,1221,1222,1223,1224,1225,1231,1232,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1251,1252,1253,1266,1267,1268,1269,1270,1271,1272,1273,1274,1281,1282,1283,1284,1296,1300,1301,1309,1310,1311,1312,1313,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1333,1334,1341,1342,1343,1344,1345,1346,1350,1351,1352,1357,1358,1359,1360,1361,1362,1363,1366,1367,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1392,1393,1402,1403,1404,1405,1406,1415,1416,1417,1418,1419,1420,1425,1426,1427,1428,1429,1434,1435,1436,1437,1438,1439,1440]
    dataset = LadickyDataset(file, trainNdxs)
    
    # Build model
    model = vgg16_model()
    
    # Parameters
    batchSize = 32
    epochs = 3
    totalBatches = ceil(dataset.size/batchSize)
    
    # Training Loop
    for epoch in range(epochs):
        print('------------------------------------------')
        start = time.perf_counter()
        for batch in range(totalBatches):
            print('*** Epoch: '+str(epoch+1)+'/'+str(epochs) +' *** Batch: '+str(batch+1)+'/'+str(totalBatches)+' ***')
            imgs, norms = dataset.get_batch(batchSize)
            loss = model.train_on_batch(imgs, norms)
            print('Loss: ' + str(loss))   
        # Saving the model
        print('Saving the model...')
        model.save('../Data/'+ args.ExperimentID + '-model.h5')
        end = time.perf_counter()
        rem = divmod((epochs-epoch-1)*(end-start),60)
        print('Remaining time: '+str(round(rem[0]))+' minute(s) and '+ str(round(rem[1]))+ ' seconds')
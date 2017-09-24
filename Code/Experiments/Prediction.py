# Imports
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
from scipy.io import savemat

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

# Prediction
def Predict(ID, Dataset, resize=False):
    
    # Load data set
    print('Loading the data set...')
    dataset = Dataset()
    
    # Load model
    print('Loading the model...')
    model = load_model('Experiments/Outputs/'+ ID + '.h5', custom_objects={'mean_dot_product': mean_dot_product, 'tf':tf})
    
    # Variables
    images = np.empty([dataset.size, dataset.batch_height, dataset.batch_width, 3], dtype=np.float32)
    normals = np.empty([dataset.size, dataset.batch_height, dataset.batch_width, 3], dtype=np.float32)
    preds = np.empty([dataset.size, dataset.batch_height, dataset.batch_width, 3], dtype=np.float32)
    
    # Prediction Loop
    print('Normal Estimation...')
    index = 0
    for i in dataset.validIndices:
        print('Index: '+str(i))
        images[index], normals[index] = dataset.get_data(i)
        preds[index] = model.predict_on_batch(images[index].reshape((1,dataset.batch_height, dataset.batch_width, 3 )))
        index += 1
    
    # Saving the result
    for i in range(dataset.size):
        img = show_image(images[i])
        norm = show_normals(normals[i])
        pred = show_normals(preds[i])
        out = Image.new('RGB', (img.size[0],3*img.size[1]))
        out.paste(img.copy())
        out.paste(norm.copy(), (0,norm.size[1]))
        out.paste(pred.copy(), (0,norm.size[1]+pred.size[1]))
        out.save('Experiments/Outputs/'+ID+'/'+str(i)+'.png')
    
    if(resize):
        # Initialisation
        Norms = np.empty([dataset.size, dataset.height, dataset.width, 3], dtype=np.float32)
        # Original normal maps
        index = 0
        for i in dataset.validIndices:
            Norms[index] = dataset.normals[i]
            index += 1
        # Resizing the predictions to the original height and width of data set
        tfSize = tf.constant([dataset.height,dataset.width], dtype=tf.int32)
        tfPreds = tf.constant(preds)
        reszPreds = tf.image.resize_images(tfPreds, tfSize)
        normPreds = tf.nn.l2_normalize(reszPreds, 2)
        # Switching to CPU for more than 2GB variables
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        sess = tf.Session(config=config)
        Preds = sess.run(normPreds)
        # Saving the results
        savemat('Experiments/Outputs/'+ ID + '.mat',{'Predictions': Preds, 'Normals': Norms})
    else:
        savemat('Experiments/Outputs/'+ ID + '.mat',{'Predictions': preds, 'Normals': normals})
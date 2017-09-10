# Imports
import tensorflow as tf
from math import ceil
from PIL import Image
import time

# Utility functions
def show_image(npimg):
    return Image.fromarray(npimg.astype(np.uint8))
def show_normals(npnorms):
    return Image.fromarray(((npnorms+1)/2*255).astype(np.uint8))

# Loss function
def mean_dot_product(y_true, y_pred):
    '''
    Arguments shape: (batchSize, height, width, components)
    '''
    dot = tf.einsum('ijkl,ijkl->ijk', y_true, y_pred) # Dot product
    n = tf.cast(tf.count_nonzero(dot),tf.float32)
    mean = tf.reduce_sum(dot) / n
    return -1 * mean

# Training
def Train(ID, Dataset, model, loss, optimizer, batchSize, epochs):
    # Load data set
    print('Loading the data set...')
    dataset = Dataset()

    # Build model
    print('Building the model...')
    model = model()
    if loss == 'mean_dot_product':
        loss = mean_dot_product
    model.compile(optimizer, loss)

    # Parameter
    totalBatches = ceil(dataset.size/batchSize)

    # Training Loop
    print('Training '+ID+'...')
    for epoch in range(epochs):
        print('------------------------------------------')
        start = time.perf_counter()
        for batch in range(totalBatches):
            print('*** Epoch: '+str(epoch+1)+'/'+str(epochs) +' *** Batch: '+str(batch+1)+'/'+str(totalBatches)+' ***')
            imgs, norms = dataset.get_batch(batchSize)
            loss = model.train_on_batch(imgs, norms)
            print('Loss: ' + str(loss))
        if( (epoch+1) % 5 == 0):
            # Saving the model
            print('Saving the model...')
            model.save('Experiments/Outputs/'+ ID + '.h5')
        # Estimating the remaining time
        end = time.perf_counter()
        rem = divmod((epochs-epoch-1)*(end-start),60)
        print('Remaining time: '+str(round(rem[0]))+' minute(s) and '+ str(round(rem[1]))+ ' seconds')
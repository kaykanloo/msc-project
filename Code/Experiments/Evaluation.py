# Imports
import json
import numpy as np
from numpy import linalg as LA

def Evaluate(ID, groundTruth, predictions):
    shape = groundTruth.shape
    # Normalization
    Norms = np.divide(groundTruth,np.reshape(LA.norm(groundTruth,axis=3), (shape[0],shape[1],shape[2],1)))
    Preds = np.divide(predictions,np.reshape(LA.norm(predictions,axis=3), (shape[0],shape[1],shape[2],1)))
    Preds = np.nan_to_num(Preds)
    # Dot product
    Dot = np.sum(np.multiply(Norms,Preds),axis=-1)
    Dot = np.clip(Dot, -1,1)
    # Error
    Err = np.rad2deg(np.arccos(Dot))
    # Stats
    Stats = {
        'Mean':float(np.mean(Err)),
        'Median':float(np.median(Err)),
        'RMSE':float(np.sqrt(np.mean(np.power(Err,2)))),
        '11.25':np.mean(np.less(Err, 11.25).astype(np.float32))*100,
        '22.5':np.mean(np.less(Err, 22.5).astype(np.float32))*100,
        '30':np.mean(np.less(Err, 30).astype(np.float32))*100,
        '45':np.mean(np.less(Err, 45).astype(np.float32))*100,}
    # Saving the results
    with open('Experiments/Outputs/'+ ID + '.eval', 'w') as fp:
        json.dump(Stats, fp)
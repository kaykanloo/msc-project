import os
import argparse
import configparser
from Experiments.Training import Train
from Experiments.Prediction import Predict
from Experiments.Evaluation import Evaluate

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("Command", help="training or prediction")
    parser.add_argument("ConfigID", help="ID corresponding to Experiments/ConfigFiles/ID.ini")
    args = parser.parse_args()
    # Reading the config File
    config = configparser.ConfigParser()
    config.read('Experiments/ConfigFiles/'+args.ConfigID+'.ini')
    # Output directory
    if not os.path.exists('Experiments/Outputs/'):
        os.makedirs('Experiments/Outputs/')
    # MAT directory
    if not os.path.exists('DataSets/MAT/'):
        os.makedirs('DataSets/MAT/')
        print('Warning: No data set is found in DataSets/MAT/')
    # Training
    if (args.Command == 'training') and ('TRAINING' in config):
        if os.path.exists('Experiments/Outputs/'+ args.ConfigID + '.h5'):
            print('Trained model (Experiments/Outputs/'+ args.ConfigID + '.h5)'+' already exists. Please (re)move this file before trying again.')
        else:
            if not os.path.exists('Experiments/Outputs/'):
                os.makedirs('Experiments/Outputs/')
            exec('from DataSets.'+config['TRAINING']['dataset']+' import Dataset')
            exec('from Models.'+config['TRAINING']['model']+' import model')
            Train(args.ConfigID, Dataset, model, config['TRAINING']['loss'], config['TRAINING']['optimizer'], int(config['TRAINING']['batchsize']), int(config['TRAINING']['epochs']))
    # Prediction
    elif (args.Command == 'prediction') and ('PREDICTION' in config):
        clean = True # No file is left from past predictions
        if os.path.exists('Experiments/Outputs/'+ args.ConfigID + '.mat'):
            clean = False
            print('Predicted normals file (Experiments/Outputs/'+ args.ConfigID + '.mat)'+' already exists. Please (re)move this file before trying again.')
        if os.path.exists('Experiments/Outputs/'+args.ConfigID+'/'):
            clean = False
            print('Predicted normal maps (Experiments/Outputs/'+ args.ConfigID + '/)'+' already exist. Please (re)move this directory before trying again.')
        if clean:
            os.makedirs('Experiments/Outputs/'+args.ConfigID+'/')
            exec('from DataSets.'+config['PREDICTION']['dataset']+' import Dataset')
            Predict(args.ConfigID, Dataset)
    # Evaluation
    elif (args.Command == 'evaluation'):
        if os.path.exists('Experiments/Outputs/'+ args.ConfigID + '.eval'):
            print('Evaluation results (Experiments/Outputs/'+ args.ConfigID + '.eval)'+' already exists. Please (re)move this file before trying again.')
        else:
            if not os.path.exists('Experiments/Outputs/'+ args.ConfigID + '.mat'):
                print('Predicted normals file (Experiments/Outputs/'+ args.ConfigID + '.mat)'+' is missing.')
            else:
                from scipy.io import loadmat
                mat = loadmat('Experiments/Outputs/'+ args.ConfigID + '.mat')
                Evaluate(args.ConfigID, mat['Normals'], mat['Predictions'])
    else:
        print("Invalid command or config file.")

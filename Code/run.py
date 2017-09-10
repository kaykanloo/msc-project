import argparse
import configparser
from Experiments.Training import Train
from Experiments.Prediction import Predict

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("Command", help="training or prediction")
    parser.add_argument("ConfigID", help="ID corresponding to Experiments/ConfigFiles/ID.ini")
    args = parser.parse_args()
    # Reading the config File
    config = configparser.ConfigParser()
    config.read('Experiments/ConfigFiles/'+args.ConfigID+'.ini')
    # Training
    if (args.Command == 'training') and ('TRAINING' in config):
        exec('from DataSets.'+config['TRAINING']['dataset']+' import Dataset')
        exec('from Models.'+config['TRAINING']['model']+' import model')
        Train(args.ConfigID, Dataset, model, config['TRAINING']['loss'], config['TRAINING']['optimizer'], int(config['TRAINING']['batchsize']), int(config['TRAINING']['epochs']))
    # Prediction
    elif (args.Command == 'prediction') and ('PREDICTION' in config):
        exec('from DataSets.'+config['PREDICTION']['dataset']+' import Dataset')
        Predict(args.ConfigID, Dataset)
    # Evaluation
    elif (args.Command == 'evaluation') and ('EVALUATION' in config):
        print('Evaluation')
    else:
        print("Invalid command or config file.")
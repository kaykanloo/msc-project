# M.Sc. Project Report
## Title: 
Surface Normal Estimation from RGB Images
## Degree: 
Master of Science in Advance Computer Science

University of Leeds, UK
## Summary:
This project explored a deep learning approach for the task of pixel-wise surface normal estimation from monocular RGB images. Initially, an analysis of the previous research was conducted. Then, data required for use in this project was obtained from two publicly available data sets. Software code was developed to compute the ground truth surface normal maps based on two
different methods and the results were used to produce the final data sets.

A deep learning pipeline was developed and a baseline network architecture was implemented. Based on the insights gained from the literature review and analysis of the state of the art methods, different modifications to the baseline model were investigated. In particular, improvements over baseline results were explored in respect to two aspects: network architectures, and the quality of the data set.

Finally, well-established evaluation metrics were implemented and the quantitative and qualitative result of evaluation were presented and compared to the state of the art methods.

## Instructions:
It is recommended to run this code on a system with at least one GPU with 12GB of memory. 

Download and install MiniConda for Python 3.6 from: https://conda.io/miniconda.html 

Install TensorFlow library by entering: conda install tensorflow-gpu

Install Keras library by : conda install keras

Install Pillow library: conda install pillow

Download a data set and put it in ./Code/DataSets/MAT/ directory. You may need to create the MAT directory. 

* Main dataset: https://s3.eu-west-2.amazonaws.com/kaykanloo-mscproject/NYUDataSet.mat

* Alternative data set: https://s3.eu-west-2.amazonaws.com/kaykanloo-mscproject/NYUAltDataSet.mat

* SUN RGB-D based data set: https://s3.eu-west-2.amazonaws.com/kaykanloo-mscproject/SUNDataSet.mat

Change your current working directory to ./Code

To train an experiment: python run.py training ConfigFileName

For produce the output of the network: python run.py prediction ConfigFileName

To evaluate the results: python run.py evaluation ConfigFileName

You can find the list of experiments in ./Code/Experiments/ConfigFiles/ directory.   

The results are stored in ./Code/Experiments/Outputs/ directory.

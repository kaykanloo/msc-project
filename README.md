# M.Sc. Project
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

* For training: python run.py training ConfigFileName

* To produce the output of the network: python run.py prediction ConfigFileName

* To evaluate the results: python run.py evaluation ConfigFileName

You can find the list of experiments in ./Code/Experiments/ConfigFiles/ directory.   

The results are stored in ./Code/Experiments/Outputs/ directory.

## Estimated surface normal maps
* Baseline model trained on main data set: https://s3.eu-west-2.amazonaws.com/kaykanloo-mscproject/Baseline.tar.gz

* VGG16 based model trained on main data set: https://s3.eu-west-2.amazonaws.com/kaykanloo-mscproject/VGG16.tar.gz

* VGG16 based model trained on SUN data set: https://s3.eu-west-2.amazonaws.com/kaykanloo-mscproject/SUNVGG16.tar.gz

## Reproducing the Data Sets (Optional)
Instead of downloading the data sets, you can use the MATLAB source codes for recomputing the surface normal maps. 

You need to donwnload the corresponding data files for each data set:

* Original NYU Depth V.2: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

* Ladicky surface normal maps: https://s3.eu-west-2.amazonaws.com/kaykanloo-mscproject/Ladicky.tar.gz

(Provided by https://www.inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip )

* SUN RGB-D Data set: http://rgbd.cs.princeton.edu/data/SUNRGBD.zip



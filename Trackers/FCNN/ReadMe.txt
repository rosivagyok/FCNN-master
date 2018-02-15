------
FCNN
------

A semester project submitted to Aalborg University, where we experiment 
with different CNNs for object trackers and fusing them togehter.

-------------
Code Access
-------------

All the code with additional benchmark code, mat files and results. With 
also the MDNet, TCNN and C-COT implementations can be found at 
[https://github.com/LouiseAbela/FCNN].

----------------
Getting Started 
----------------

To run a simple test with the tracker

1. Run setup.m
2. Run demo.m

This will run an example of the tracker on Deer dataset, and outputs 
a video 'Deer.avi' and the bounding boxes in a text file 'Deer.txt'.
To run it on other sequences download the sequence put it under the correct
dataset folder, if it's another dataset then OTB, change the appropriate path
in 'genConfig.m' 

----------------
Training
----------------

To train the network:

1. Run setup.m
2. Download 'VOT2016' dataset
3. Delete 'car1' from the list 
4. Change the variables 'opts.seqList' and 'opts.dataPath' to the path
   of the dataset and the dataset list in fusion_demo.m
5. Change any options like batch_size and number of epochs in 'second_cnn_train_dag.m'
6. Run 'fusion_demo.m'

This will display a graph with the error and objective and will update with every epoch

----------------
Tracking
----------------

To use the trained network:

1. Run setup.m
2. Set the path to the new model in 'run_fusion.m'
3. Use 'demo.m'
   OR
   Run the benchmark which will use 'run_FCNN'

--------------
Prerequisites
--------------

1. Matlab R2016b, with these additional libraries:
    - MatConvNet

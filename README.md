# 2021-California-State-Science-Fair
This repository contains code used to collect data for the 2021 California State Science Fair. The code contains little or no comments: it contains the exact code that I used to gather data. 

[CS231n Assignments](https://github.com/MaximumLimits/2021-California-State-Science-Fair/tree/main/CS231n%20Assignments) contains the assignments I completed while I took the Stanford course [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/). These assignmented helped me fully understand the mathematics behind neural networks, including matrix operations, derivatives, training, optimization techniques, and loss evaluation.

The raw photos and image dataset is not included in this repository. The datasest was created by Kaggle user _cchangcs_. (It can be downloaded at https://www.kaggle.com/asdasdasasdas/garbage-classification.)

_image.py_ is used to save the raw photos from the image dataset as NumPy arrays. The file also contains code for viewing, preprocessing, augmenting, and loading the NumPy arrays. 

_pretest.py_ contains a two-layer perceptron neural network containing two fully-connected layers. I created the neural network only with NumPy without machine learning libraries while I took the aforementioned Stanford course. _pretest.py_ was created to debug _image.py_ and determine whether I needed to learn more about neural networks before starting on this science project. 

_create greaph with everything.py_ contains the code for all the models and the graph that displays the 7 most significant ones. 

_fine tuning.py_ was initially used to fine-tune the hyperparameters of a previous iteration of TARnet, such as learning rate and batch size. However, after TARnet's architecture was changed, hyperparameters were manually chosen and evaluated.

_final.py_ contains code for creating, training, and evaluating the final iteration of TARnet. 

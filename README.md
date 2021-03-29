# 2021-California-State-Science-Fair
This repository contains code used to collect data for the 2021 California State Science Fair. The code contains little or no comments: it contains the exact code that I used to gather data.

"image.py" is used to save the raw photos from the dataset created by Kaggle user cchangcs (https://www.kaggle.com/asdasdasasdas/garbage-classification) as numpy arrays. The file also contains code for viewing, preprocessing, and loading the numpy arrays. 

"pretest.py" contains a two-layer perceptron neural network containing two fully-connected layers. I created the neural network only with NumPy without machine learning libraries while I took the Stanford course _CS231n: Convolutional Neural Networks for Visual Recognition_ (http://cs231n.stanford.edu/2017/). Creating this network helped me fully understand the mathematics behind neural networks, including matrix operations, derivatives, and loss evaluation. "pretest.py" was created to debug "image.py" and determine whether I needed to study more before starting on this science project. 

"create greaph with everything.py" contains the code for all the models and the graph that displays the 7 most significant ones. 

"fine tuning" was initially used to fine-tune the hyperparameters of TARnet, such as learning rate and batch size. However, after TARnet's architecture was changed, hyperparameters were manually chosen and evaluated.

"final.py" contains code for creating, training, and evaluating the final iteration of TARnet. 

# 2021-California-State-Science-Fair
This repository contains code used to collect data for the 2021 California State Science Fair. The code contains little or no comments: it contains the exact code that I used to gather data. 

[_CS231n Assignments_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/tree/main/CS231n%20Assignments) contains the assignments I completed while I took the Stanford course [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/2017/). These assignmented helped me fully understand the mathematics behind neural networks, including matrix operations, derivatives, training, optimization techniques, and loss evaluation.

The raw photos and image dataset is not included in this repository. The datasest was created by Kaggle user _cchangcs_. (It can be downloaded at https://www.kaggle.com/asdasdasasdas/garbage-classification.)

[_images.py_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/blob/main/images.py) is used to save the raw photos from the image dataset as NumPy arrays. The file also contains code for viewing, preprocessing, augmenting, and loading the NumPy arrays. 

[_pretest.py_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/blob/main/pretest.py) contains a two-layer perceptron neural network containing two fully-connected layers. I created the neural network only with NumPy without machine learning libraries while I took the aforementioned Stanford course. _pretest.py_ was created to debug _image.py_ and determine whether I needed to learn more about neural networks before starting on this science project. 

[_create greaph with everything.py_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/blob/main/create%20graph%20with%20everything.py) contains the code for all the models and the graph that displays the 7 most significant ones. The training accuracies, training losses, valiation accuracies, and validation losses of these models are stored in the folders [_training_accuracies_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/tree/main/training_accuracies), [_training_losses_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/tree/main/training_losses), [_validation_accuracies_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/tree/main/validation_accuracies), and [_validation_losses_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/tree/main/validation_losses) as NumPy array (.npy) files.

[_fine tuning.py_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/blob/main/fine%20tuning.py) was initially used to fine-tune the hyperparameters of a previous iteration of TARnet, such as learning rate and batch size. However, after TARnet's architecture was changed, hyperparameters were manually chosen and evaluated.

[_final.py_](https://github.com/MaximumLimits/2021-California-State-Science-Fair/blob/main/final.py) contains code for creating, training, and evaluating the final iteration of TARnet. 

# 2021-California-State-Science-Fair
This repository contains code used to collect data for the 2021 California State Science Fair. The code contains little or no comments: it contains the exact code that I used to gather data.

"image.py" is used to save the raw photos from the dataset created by Kaggle user cchangcs (https://www.kaggle.com/asdasdasasdas/garbage-classification) as numpy arrays. The file also contains code for viewing, preprocessing, and loading the numpy arrays. 

"create greaph with everything.py" contains the code for _all_ the models and the graph that displays the 7 most significant ones. 

"fine tuning" was initially used to fine-tune the hyperparameters of TARnet, such as learning rate and batch size. However, after TARnet's architecture was changed, hyperparameters were manually chosen and evaluated.

"final.py" contains code for creating, training, and evaluating TARnet. 

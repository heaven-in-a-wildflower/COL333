# Assignment 3 (Part I): Learning with Neural Networks

# Problem Statement

We are given a data set consisting of images of different bird species with a total of K = 10  bird species present in the data set. Each species having between 500 and 1,200 images in the data set and each image containing a type single bird only. Please design a neural network that takes as input a bird image and predicts the class label (one of the K possible labels) corresponding to each image. The next section will provide some hints on designing and validating your model. 


### Visualising Class Activation Maps

- To gain insights into which features the model is learning, we can apply **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Grad-CAM helps visualize the important regions in an image that contribute to the model's decision for the predicted class by highlighting areas the model focuses on. Once the model is trained, apply Grad-CAM to see which regions of the image are most influential for each predicted class.
- One of the most popular implementation is : [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

# Training and Evaluation process

**Training Code**

- [bird.py](http://part1.py) should contain the all the main code for train and testing the model.  We will run bird.py for training and testing in the following manner.
- Do  not change this format.
    
    ```python
    # Running code for training. save the model in the same directory with name "**bird.pth"**
    python bird.py path_to_dataset train bird.pth 
    
    # Running code for inference
    python bird.py path_to_dataset test bird.pth
    ```
# Starter Code

### Getting Started

Started code is available on the Moodle and have following structure:

```python
A3
|- enviroment.yaml #contains all the required libraries 
|- install.sh # install all the dependencies.
|- bird.py # Add all your code here.
|- run.ipynb # Contains instructions to run the code. 
|- Dataset # Download from kaggle 
```

> Check run.ipynb to setup the environment and dataset.
> 

## Development Environment

You can develop your solution either using your local setup or Kaggle. We recommend using Kaggle as it provide GPUs that can be helpful in model training. 

- Local setup: You can create a Python environment on your machine with `conda` using the command `conda env create -f environment.yml` . Here, `environment.yml` file is available with the starter code.
- You can setup kaggle following the instructions here:

Dataset link is provided in the document below. You can directly import the dataset to Kaggle using name [**Identify-the-Birds**](https://www.kaggle.com/datasets/aayushkt/identify-the-birds)

[instructionsA3.pdf](Assignment%203%20(Part%20I)%20Learning%20with%20Neural%20Network%2014d43fc51f6680e497f6ecfa2da55887/instructionsA3.pdf) 
****
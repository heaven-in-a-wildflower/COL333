# Assignment 3 (Part I): Learning with Neural Networks

## Problem Statement

We are given a data set consisting of images of different bird species with a total of K = 10  bird species present in the data set. Each species having between 500 and 1,200 images in the data set and each image containing a type single bird only. The task is to design a neural network that takes as input a bird image and predicts the class label (one of the K possible labels) corresponding to each image.

## Visualising Class Activation Maps

- To gain insights into which features the model is learning, we can apply **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Grad-CAM helps visualize the important regions in an image that contribute to the model's decision for the predicted class by highlighting areas the model focuses on. Once the model is trained, apply Grad-CAM to see which regions of the image are most influential for each predicted class.
- One of the most popular implementation is : [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

## Training and Evaluation process
    # Running code for training. save the model in the same directory with name "**bird.pth"**
    python bird.py path_to_dataset train bird.pth 
    
    # Running code for inference
    python bird.py path_to_dataset test bird.pth
- Dataset link is provided [Here](https://drive.google.com/drive/folders/1CLxNjtoLfV9e678kXr_21wiD_yoxpkDZ?usp=sharing)

# Assignment 3 (Part II): Representation Learning

## Problem Description

- The goal of the assignment is learning structured latent representations from unlabelled images of “digits” to enable classification with minimal examples. Specifically, learning compact latent representation from unlabelled images and then use these to build a model that categorises images of digits into one of three groups (1, 4, or 8) with just 2-3 examples!
- Formally, this part of the assignment has two components. First, to train a Variational Autoencoder (VAE) to extract useful features from the images. Then, to use the learnt features for a downstream task like classification in a low-data regime. In particular, we implement a Gaussian Mixture Model (GMM) to cluster the extracted latent representations. Once the clusters are learned,  we can categorise any unseen images by first extracting their features through the trained encoder and assigning a cluster label based on the maximum likelihood.

## Training and Evaluation
-Please see report for implementation details.
```python
# Running code for training. save the model in the same directory with name "vae.path**"
# Save the GMM parameters in the same folder. You can use pickle to save the parameters.** 
python vae.py path_to_train_dataset path_to_val_dataset train vae.pth gmm_params.pkl

# Running code for vae reconstruction.
# This should save the reconstruced images in numpy format. see below for more details.
python vae.py path_to_test_dataset_recon test_reconstruction vae.pth

#Running code for class prediction during testing
python vae.py path_to_test_dataset test_classifier vae.pth gmm_params.pkl
```

- Dataset link is provided [Here](https://drive.google.com/drive/folders/1CLxNjtoLfV9e678kXr_21wiD_yoxpkDZ?usp=sharing)

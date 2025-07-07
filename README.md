


---
title: European Mushroom Common Genus Image Classifier
emoji: 📊
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: apache-2.0
---

# Northern European Mushroom Genus Classification Model for Poisonous Species Detection

## Description

This repository contains a series of Python scripts and Jupyter Notebooks aimed at developing and deploying a convolutional neural network (CNN) model to classify images of common Northern European mushroom genera. The project includes data cleaning, model training, and a web-based interface for image classification. Below are the details of the key components.

Which genera were used to train the model?

## Genera:
Amanita
Boletus
Hygrocybe
Entoloma
Russula
Agaricus
Lactarius
Webcaps
Cortinarius
Slippery Jacks
Suillus

## Key Features
Classifies common Northern European mushroom genera using image data.
Includes thorough image cleaning and pre-processing steps for improved accuracy.
Provides clear instructions for running the model.

## Target Audience
Individuals concerned with safe mushroom foraging in Northern Europe.
Machine learning enthusiasts interested in image classification tasks.
Researchers working on mushroom identification and toxicity detection.



## Overview of 1_mushshrooms-cnn-clasification-cleaning.py

The file "1_mushshrooms-cnn-clasification-cleaning.py" is a Jupyter Notebook written in Python that focuses on building and refining a CNN model for classifying images of common Northern European mushroom genera. The dataset, sourced from Kaggle (https://www.kaggle.com/maysee/mushrooms-classification-common-genuss-images), contains 300 to 1500 images per genus, with nine distinct genera: Agaricus, Suillus, Entoloma, Hygrocybe, Amanita, Lactarius, Russula, Boletus, and Cortinarius. The primary objective is to explore the dataset, experiment with data augmentation and sampling techniques, develop a mushroom image classifier, and identify images with the highest errors for potential dataset cleaning.

The notebook utilizes the Fastai library along with other tools like PyTorch Image Models (timm), scikit-learn, and Matplotlib for data handling, model training, and visualization. It begins by setting up the environment, defining paths for training and validation data, and analyzing the dataset's image distribution across the genera. The code includes data preprocessing steps, such as splitting the dataset into training and validation sets, and employs a pre-trained ConvNeXt Tiny model for classification.

Key sections of the notebook involve training the model, evaluating its performance, and using the ImageClassifierCleaner tool to manually review and correct misclassified images. Visualization of top losses and confusion matrices helps identify problematic images and genera with frequent misclassifications, such as Lactarius and Russula. The notebook is executed in a Kaggle environment with GPU acceleration, and the results include graphical outputs of loss distributions and confusion metrics.

This work serves as a foundational step for improving the accuracy of mushroom genus classification and cleaning the dataset for future modeling efforts.

##  Overview of 2.Mushrooms-genus-cnn-modeling.py

The file "2.Mushrooms-genus-cnn-modeling.py" is a Jupyter Notebook written in Python that aims to develop and evaluate different CNN models to classify images of the most common Northern European mushroom genera. The dataset, derived from a cleaned version hosted on Kaggle, includes images of nine genera: Agaricus, Suillus, Entoloma, Hygrocybe, Amanita, Lactarius, Russula, Boletus, and Cortinarius, with varying image counts ranging from 288 to 1410 per genus.

The notebook leverages the Fastai library, along with PyTorch Image Models (timm), scikit-learn, and Matplotlib, to handle data preprocessing, model training, and visualization. It starts by setting up the environment, defining paths for training and validation data, and analyzing the dataset's distribution. The data is split into training (85%) and validation (15%) sets using stratified sampling to maintain class balance, with images copied to respective directories.

Key steps include verifying image integrity, defining a DataBlock for data loading with random resizing, and experimenting with models like ConvNeXt. The notebook trains the model, evaluates its performance (achieving an error rate of 0.026, macro average precision of 0.98, and recall of 0.97 with the best model, convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384), and exports the trained model. Visualizations and the ImageClassifierCleaner tool are used to identify misclassifications, with suggestions for improvement including gathering more data, balancing the dataset, and using larger image sizes for better resizing.

## Overview of app.py

The file "app.py" is a Python script that creates a web-based interface using Gradio to classify images of common Northern European mushroom genera. It utilizes a pre-trained Fastai model, specifically a ConvNeXt model saved as 'ConvNext_RmsProps.pkl', to predict the genus of uploaded mushroom images. The script defines a learner object by loading the model and extracts the labels (nine genera: Entoloma, Suillus, Hygrocybe, Agaricus, Amanita, Lactarius, Russula, Boletus, and Cortinarius) from the model's data loader.

The classify function processes an input image by converting it into a PILImage, running it through the learner to get predictions, and returning a dictionary of probabilities for each genus. The Gradio interface is configured with this function, accepting an image input and outputting a label with the predicted genus and its probability. It includes examples of various mushroom images and provides a title and description detailing the model's purpose and scope.

The interface is launched locally without sharing, offering an interactive tool for users to upload images and receive classifications based on the trained model.

## After Modeling We Have Found
Best performing model: convnext_tiny_in22k
Error rate: 0.036
Macro avg: precision 0.96 and recall 0.96
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

##  Project Workflow
Data Cleaning and Initial Modeling: Start with "1_mushshrooms-cnn-clasification-cleaning.py" to preprocess the dataset, train an initial model, and clean misclassified images.
Model Optimization: Use "2.Mushrooms-genus-cnn-modeling.py" to experiment with different models and parameters, ultimately selecting and exporting the best-performing model.
Deployment: Deploy the trained model via "app.py" to create an accessible web interface for classifying new mushroom images.

## Future Improvements
Gather more data to enhance model robustness.
Balance the dataset to address class imbalance issues.
Use larger image sizes for improved resizing and better feature extraction.

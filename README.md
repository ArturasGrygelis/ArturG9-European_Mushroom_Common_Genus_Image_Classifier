


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

Description:
This repository presents a machine learning model designed to help prevent mushroom poisoning by accurately classifying the genus of common Northern European mushrooms based on image data. It emphasizes the crucial step of cleaning and pre-processing images to enhance model performance.

Which genera were used to train the model?

## Genera:

Amanita
Boletus
Hygrocybe
Entoloma
Russula
Agaricus
Lactarius
Webcaps Cortinarius
Slippery Jacks Suillus

## Key Features:

Classifies common Northern European mushroom genera using image data.
Includes thorough image cleaning and pre-processing steps for improved accuracy.
Provides clear instructions for running the model.

## Target Audience:

Individuals concerned with safe mushroom foraging in Northern Europe.
Machine learning enthusiasts interested in image classification tasks.
Researchers working on mushroom identification and toxicity detection.

* After modeling we have found:
* best performing model: convnext_tiny_in22k 
* error rate: 0.036 
* Macro avg:  precision 0.96 and recall 0.96 



Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

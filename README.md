# PyTorch Implementation of TEAM (TensorFlow to PyTorch Conversion)

This repository provides a PyTorch implementation of the TEAM model, originally developed in TensorFlow. The purpose of this project is to convert the original TensorFlow codebase into PyTorch to facilitate model training and experimentation using the PyTorch framework.


## Original Work

The TEAM model was introduced in the following paper:  
**Title:** *The transformer earthquake alerting model: a new versatile approach to earthquake early warning*  
**Authors:** Jannes Münchmeyer, Dino Bindi, Ulf Leser, Frederik Tilmann 
**Published in:** *Geophysical Journal International*  
**Link to the paper:** [TEAM Paper](https://doi.org/10.1093/gji/ggaa609)

The original TensorFlow implementation can be found at this GitHub repository:  
[Original TensorFlow Code](https://github.com/yetinam/TEAM)

## Conversion to PyTorch

The goal of this project is to ensure the TEAM model functions correctly in PyTorch, maintaining its original accuracy and performance. We aim to make the codebase more accessible to researchers and developers who prefer using PyTorch for deep learning models.

## Additional Evaluation Method

In addition to the original evaluation methods used in the paper, we have introduced an extended evaluation process. Specifically, we added an evaluation that includes all stations for each event, allowing for a more comprehensive assessment of the model's performance across different conditions and datasets.


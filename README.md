# Potato Disease Detection Using CNN

## Overview
This repository contains a Jupyter Notebook (`ECNN_ADAMAX.ipynb`) that implements a Convolutional Neural Network (CNN) for detecting diseases in potato leaves. The project leverages deep learning techniques to analyze images and classify the health status of potato plants, helping to improve agricultural practices and crop yield.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV

## Dataset
The dataset consists of images of potato leaves, categorized into different classes representing healthy and diseased leaves. Plant Village dataset consists of more than 10,000 images of leaves of different species. This data is essential for training the CNN model and can be obtained from [[Dataset Link](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)].

## Features
- Data loading and preprocessing
- Model architecture definition using CNN
- Training with the Adamax optimizer
- Evaluation of model performance
- Visualization of training metrics

## Getting Started
### Prerequisites
To run this notebook, you need:
- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Potato-Disease-Detection-Using-CNN.git

2. Navigate to the project directory:
   ```bash
   cd Potato-Disease-Detection-Using-CNN

3.(Optional) Create a virtual environment:
   ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
4.(Optional) If you don't have requirements.txt, you can manually install the necessary libraries:
   ```bash
  pip install tensorflow keras numpy matplotlib opencv-python
  ```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook ECNN_ADAMAX.ipynb

2. Follow the instructions within the notebook to load the dataset, train the model, and visualize the results. Ensure that the dataset is placed in the specified directory as mentioned in the notebook.

## Results
The model's performance can be evaluated using various metrics such as accuracy, precision, recall, and F1-score. The Jupyter Notebook contains visualizations of the training process, including graphs for loss and accuracy over epochs. Additionally, a classification report detailing the model's performance on the test set is generated, allowing for a comprehensive analysis of its effectiveness in detecting potato diseases.


## Acknowledgments
- TensorFlow and Keras for providing powerful tools for deep learning.
- The contributors and researchers in the field of plant disease detection.

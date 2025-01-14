# Landmark Classification & Tagging for Social Media

## Project Overview

Photo sharing and storage platforms often benefit from location data to enhance user experiences through automatic tagging and photo organization. However, many images lack location metadata, either due to missing GPS data or metadata removal for privacy reasons. To infer location information, identifying landmarks in images becomes crucial.

This project tackles this problem by building models to predict the location of images based on visible landmarks. It follows the complete machine learning pipeline, including data preprocessing, CNN design and training, transfer learning, and evaluation. The project aligns with the rubric requirements of the "AWS AI & ML Scholarship" program.

Examples from the landmarks dataset include iconic locations such as Death Valley, the Brooklyn Bridge, and the Eiffel Tower.

As part of the AWS AI & ML Scholarship

---

## Features
- Training a CNN from scratch to classify landmarks with custom architecture.
- Leveraging transfer learning with pre-trained CNN models for improved accuracy.
- Comprehensive data preprocessing, augmentation, and visualization.
- Model training, validation, and testing with hyperparameter optimization.
- Exporting trained models using Torch Script for deployment.
- Development of an app to load TorchScript models and make predictions.

---

## Project Steps

### Step 1: Training a CNN Model from Scratch
- Built a Convolutional Neural Network (CNN) for landmark classification.
- Visualized and prepared the dataset for training using data loaders.
- Data preprocessing included resizing, cropping, normalization, and augmentation.
- Exported the best-performing network using Torch Script.

**Jupyter Notebook**: `cnn_from_scratch.ipynb`

### Step 2: Using Transfer Learning
- Established a CNN for landmark classification using transfer learning.
- Explored multiple pre-trained models and selected the most suitable one.
- Trained and tested the transfer-learned network, achieving at least 60% test accuracy.
- Exported the optimal transfer learning solution using Torch Script.

**Jupyter Notebook**: `transfer_learning.ipynb`

### Step 3: Building the Application
- Developed an app to load the TorchScript-exported models.
- Processed new input images and displayed predictions.
- Ensured functionality for unseen data, as demonstrated in `app.ipynb`.

---

## How to Navigate the Project

### `src` Folder
- Contains individual Python files for each function implementation and their corresponding test cases:
  - `train.py`: Training and validation logic.
  - `model.py`: CNN architecture implementation.
  - `data.py`: Data loading and preprocessing utilities.
  - `helpers.py`: Helper functions for visualization and logging.
  - `predictor.py`: TorchScript-based prediction logic.
  - `transfer.py`: Transfer learning model setup.
  - `optimization.py`: Loss functions and optimizer setup.

### Jupyter Notebooks
- **`cnn_from_scratch.ipynb`**: Training a CNN model from scratch.
- **`transfer_learning.ipynb`**: Implementing and training a transfer learning model.
- **`app.ipynb`**: Running the trained models on new data through a simple app interface.

---

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/AdrianCuevasML/Landmark_Classification_and_Tagging_for_Social_Media.git
```

### Navigate to the Repository Directory
```bash
cd landmark-classification
```

## Acknowledgments
This was the third project of the "Udacity Machine Learning Fundamentals Nanodegree" offered by AWS as part of the "AWS AI & ML Scholarship." The project follows the rubric requirements, ensuring all deliverables are met.

For any questions or feedback, feel free to contact me!

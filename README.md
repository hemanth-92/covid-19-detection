# **Early Detection of Cough-Related Diseases Using COVID-19 Audio Data**

## **Overview**

COVID-19, first identified in Wuhan, China, in December 2019, rapidly evolved into a global pandemic caused by the SARS-CoV-2 virus. This project aims to develop a cough sound classification model using COVID-19 audio data to enable early detection of the disease.

## **Project Goals**

The primary objectives of this project are:

- Develop a machine learning model to classify cough sounds for COVID-19 detection.
- Explore various audio feature extraction techniques.
- Address data imbalance challenges within the dataset.
- Optimize model performance for better accuracy.

## **Team Members**

- **S. Amarnath**  
- **A. Hemanth**  
- **M. Sai Dhanush**  

## **Project Supervisors**

- **Dr. P. Penchala Prasad** (Associate Professor, CSE-DS)  
- **Mr. G. Vikram Chandra (M.Tech, PhD)** (Assistant Professor, CSE-DS)  

## **Individual Contributions**

- **Amarnath:** Feature extraction, exploratory data analysis (EDA), and training multiple models (KNN, XGBoost, Random Forest, CNN, and TD-CNN+BiLSTM).  
- **Hemanth:** Executed models on AWS, contributed to data augmentation techniques (SMOTE, Gaussian noise), and worked on the TD-CNN+BiLSTM model.  
- **Dhanush:** Conducted research, proposed SMOTE and advanced feature extraction techniques, and suggested improvements for model performance.  

---

## **Methodology**

The project follows a structured approach, including data collection, feature extraction, data preprocessing, model development, and evaluation.

### **1. Data Collection**

- **Source:** Audio data obtained from Kaggle.
- **Data Type:** COVID-19 cough sound recordings.
- **File Formats:**
  - `.wav` files (27.6K)
  - `.json` files (27.6K)
  - `.csv` metadata file (1)
- **Data Loading Modules:**
  - `pip install kaggle`
  - `pip install opendatasets`
- **Download Command:**
  - `op.download("kaggle dataset link")`

### **2. Feature Extraction**

Eleven different audio features were extracted using the `Librosa` module:

- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Chroma**
- **Mel-Spectrogram**
- **Spectral Contrast**
- **Tonnetz**
- **Spectral Centroid**
- **Spectral Roll-off**
- **Zero-Crossing Rate (ZCR)**
- **Spectral Bandwidth**
- **RMSE (Root Mean Square Energy)**
- **Spectral Flatness**

### **3. Data Imbalance Handling**

To address data imbalance, the following techniques were used:

- **Data Augmentation:**
  - Gaussian Noise: `Xnoisy = X + N(μ, σ^2)`
- **SMOTE (Synthetic Minority Oversampling Technique):**
  - Generates synthetic samples to balance class distribution.

### **4. Model Selection and Training**

- **Model:** TimeDistributed CNN + Bi-LSTM
- **Optimization:** Adam Optimizer

#### **Adam Optimization Algorithm:**

1. Initializes parameters, first and second moment estimates.
2. Computes gradients of the loss function.
3. Updates moving averages of gradient and squared gradient.
4. Applies bias correction.
5. Updates parameters using corrected estimates with adaptive learning rate.
6. Iterates until convergence.

#### **TimeDistributed CNN:**

- Applies a CNN layer (like Conv2D) across the time dimension.
- Helps capture sequential patterns in audio data.

#### **Bi-LSTM (Bidirectional Long Short-Term Memory):**

- Processes data both forward and backward.
- Captures full temporal dependencies for better classification accuracy.

### **5. Model Evaluation**

Evaluation metrics used to assess model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

## **Objectives and Contributions**

### **1. Feature Selection**

- **Objective:** Identify and select key audio features to enhance model efficiency.
- **Contribution:** Extracting MFCC, Chroma, Mel-Spectrogram, Spectral Contrast, and other relevant features.

### **2. Addressing Data Imbalance**

- **Objective:** Mitigate the impact of class imbalance.
- **Contribution:** Implementing SMOTE and data augmentation techniques such as Gaussian noise.

### **3. Model Optimization**

- **Objective:** Improve training and classification accuracy.
- **Contribution:** Implementing the Adam optimization algorithm for faster convergence and better learning.

---

## **Results**

- **Batch Accuracy:** 88.50%
- **Training Accuracy:** 94.54%
- **Validation Accuracy:** 76.70%

### **Performance Visualizations:**

- **Accuracy Graph:** *(Insert Image)*
- **ROC Curve:** *(Insert Image)*

---


## **Conclusion**

This project successfully explores cough-based COVID-19 detection using deep learning. While promising results were achieved, future work can focus on refining the dataset and improving classification techniques for more reliable early disease detection.

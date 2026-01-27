#DATASET LOADER

#This file contains functions to load different datasets for our KNN experiments
#We use 5 datasets of different sizes to test how our model performs:
#2 small datasets (Iris, Wine)
#2 medium datasets (Digits, Breast Cancer)
#1 large dataset (Nursery from UCI)


#IMPORTING LIBRARIES

#numpy - Library for numerical operations and working with arrays
import numpy as np

#os - Python's built-in library for operating system functions
#We use it to check if files exist on the computer
import os

#sklearn.datasets - Contains several famous datasets built into sklearn
#These are ready to use without downloading anything
#load_iris: 150 flower samples, 4 features, 3 classes
#load_wine: 178 wine samples, 13 features, 3 classes
#load_digits: 1797 handwritten digit images, 64 features, 10 classes
#load_breast_cancer: 569 tumor samples, 30 features, 2 classes
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer

#pandas - Library for data manipulation and reading files
#We use it to load the Nursery dataset from a CSV file
import pandas as pd


#LOAD DATASET FUNCTION

#This is the main function that loads any dataset by name. It returns the features (X), labels (Y), and information about the dataset
def load_dataset(dataset_name):
    
    #dataset_name is a string like 'iris', 'wine', 'digits', 'breast_cancer', or 'nursery'. We check which dataset was requested and load it accordingly
    
    #IRIS DATASET - small
    
    if dataset_name == 'iris':
        
        #The Iris dataset is one of the most famous datasets in machine learning
        #It contains measurements of 150 iris flowers from 3 different species
        #Features: sepal length, sepal width, petal length, petal width
        #Classes: setosa, versicolor, virginica
        
        #load_iris() returns a special object with data, target, feature_names, etc.
        data = load_iris()
        
        #data.data contains the feature values (150 samples x 4 features)
        X = data.data
        
        #data.target contains the class labels (0, 1, or 2 for each sample)
        Y = data.target
        
        #Names of the features and classes (useful for understanding results)
        feature_names = data.feature_names
        target_names = data.target_names
        
        #Create a dictionary with information about this dataset
        info = {
            'name': 'Iris',
            'n_samples': X.shape[0],      #Number of rows (150)
            'n_features': X.shape[1],     #Number of columns (4)
            'n_classes': len(np.unique(Y)), #Number of unique classes (3)
            'size_category': 'small'
        }
    
    
    #WINE DATASET - small
    
    elif dataset_name == 'wine':
        
        #The Wine dataset contains chemical analysis of wines from 3 different cultivars
        #178 samples with 13 features each
        #Features include things like alcohol content, malic acid, ash, etc.
        #Classes: three different wine cultivars (types of grape)
        
        data = load_wine()
        X = data.data
        Y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
        info = {
            'name': 'Wine',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(Y)),
            'size_category': 'small'
        }
    
    
    #DIGITS DATASET - medium
    
    elif dataset_name == 'digits':
        
        #The Digits dataset contains images of handwritten digits (0-9)
        #Each image is 8x8 pixels, flattened into 64 features
        #1797 samples, 10 classes (digits 0 through 9)
        #This is a simplified version of the famous MNIST dataset
        
        data = load_digits()
        X = data.data
        Y = data.target
        
        #Digits doesn't have meaningful feature names, so we create generic ones. Each feature represents one pixel in the 8x8 image
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        target_names = data.target_names
        
        info = {
            'name': 'Digits (8x8)',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(Y)),
            'size_category': 'medium'
        }
    
    
    #BREAST CANCER DATASET - medium
    
    elif dataset_name == 'breast_cancer':
        
        #The Breast Cancer Wisconsin dataset contains tumor measurements
        #569 samples with 30 features each
        #Features are computed from digitized images of cell nuclei
        #Classes: malignant (cancerous) or benign (not cancerous)
        #This is a binary classification problem (2 classes)
        
        data = load_breast_cancer()
        X = data.data
        Y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
        info = {
            'name': 'Breast Cancer Wisconsin',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(Y)),
            'size_category': 'medium'
        }
    
    
    #NURSERY DATASET - large 
    
    elif dataset_name == 'nursery':
        
        #The Nursery dataset is from the UCI Machine Learning Repository
        #It contains data about nursery school applications
        #12960 samples with 8 features
        #This is our largest dataset to test how our KNN scales
        
        #Unlike sklearn datasets, this one needs to be downloaded separately
        #We need to find the file on the computer
        
        print("  Loading Nursery dataset from UCI...")
        
        #List of possible locations where the file might be
        #We check multiple paths because different users might store it differently
        possible_paths = [
            'datasets/nursery.data',
            'data/nursery.data',
            '../datasets/nursery.data',
            './nursery.data'
        ]
        
        #Try to find the file
        data_path = None
        for path in possible_paths:
            #os.path.exists() returns True if the file exists
            if os.path.exists(path):
                data_path = path
                break
        
        
        #Load the data using pandas
        #header=None because the file has no header row
        data = pd.read_csv(data_path, header=None)
        
        #Split into features and target
        #iloc[:, :-1] means "all rows, all columns except the last one"
        X_raw = data.iloc[:, :-1].values
        
        #iloc[:, -1] means "all rows, only the last column"
        Y_raw = data.iloc[:, -1].values
        
        #The Nursery dataset has categorical (text) values, not numbers
        #KNN needs numbers, so we need to convert text to numbers
        #LabelEncoder converts categories like ['low', 'med', 'high'] to [0, 1, 2]
        from sklearn.preprocessing import LabelEncoder
        
        #Convert each feature column from text to numbers
        X = np.zeros_like(X_raw, dtype=float)
        feature_names = []
        
        for i in range(X_raw.shape[1]):
            #Create a new encoder for each column
            le = LabelEncoder()
            #fit_transform learns the categories and converts them to numbers
            X[:, i] = le.fit_transform(X_raw[:, i])
            feature_names.append(f'feature_{i}')
        
        #Convert target column from text to numbers
        le_target = LabelEncoder()
        Y = le_target.fit_transform(Y_raw)
        target_names = le_target.classes_
        
        info = {
            'name': 'Nursery (UCI)',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(Y)),
            'size_category': 'large'
        }
    
    
    #Return all the data and information
    return X, Y, feature_names, target_names, info


#GET AVAILABLE DATASETS FUNCTION

#This function returns a list of all dataset names we can use
#It's used by experiments.py to know which datasets to loop through

def get_available_datasets():
    
    #We have 5 datasets total:
    #Small: iris (150 samples), wine (178 samples)
    #Medium: digits (1797 samples), breast_cancer (569 samples)
    #Large: nursery (12960 samples)
    
    return ['iris', 'wine', 'digits', 'breast_cancer', 'nursery']
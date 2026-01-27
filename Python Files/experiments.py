#EXPERIMENTS FILE

#In this script we compare our own implementation of the K-Nearest Neighbours classifier with sklearn's built-in KNN implementation(implemented and ran on this file).
#The experiments are run on multiple datasets of different sizes.

#For each dataset we vary: 1- The training set size 2- The number of neighbours (k) 3- The distance metric 4- The weighting scheme

#We record both machine learning performance metrics and computational metrics.
#Results are saved to separate CSV files for each implementation:
#customKNN_results.csv
#sklearnKNN_results.csv



#IMPORTING LIBRARIES

#time - Python's built-in library for measuring how long code takes to run. We use it to measure training and prediction times
import time

#csv - Python's built-in library for reading and writing CSV files. We use it to save our experimental results
import csv

#sys - Python's built-in library for system-related functions. We use it to measure memory usage of our data
import sys

#sklearn.metrics - Contains functions to evaluate how well our model performs
#accuracy_score: percentage of correct predictions
#precision_score: of all positive predictions, how many were correct
#recall_score: of all actual positives, how many did we find
#f1_score: harmonic mean of precision and recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#KNeighborsClassifier - sklearn's optimized KNN implementation. We compare our custom implementation against this
from sklearn.neighbors import KNeighborsClassifier

#train_test_split - Function to split data into training and testing sets. Randomly divides data while keeping class proportions balanced
from sklearn.model_selection import train_test_split

#StandardScaler - Function to standardize features (z-score normalization). Makes all features have mean=0 and standard deviation=1
from sklearn.preprocessing import StandardScaler

#Importing our own custom KNN model that we built from scratch
from customKNN_Model import customKNN_Model

#Importing functions to load our datasets(functions implemented in seperate files(on the dataset_loader.py file))
#load_dataset: loads a specific dataset by name
#get_available_datasets: returns list of all dataset names we can use
from dataset_loader import load_dataset, get_available_datasets


#EXPERIMENT CONFIGURATION

#These are the settings we will test in our experiments. By trying different combinations, we can see how each setting affects performance

#k_Range - Different values of k (number of neighbours) to test
k_Range = [1, 5, 10, 20, 50, 100]

#distanceMetrics - Different ways to measure distance between points
#euclidean = straight line distance (Pythagorean theorem)
#manhattan = distance along grid lines
distanceMetrics = ["euclidean", "manhattan"]

#votingTypes - different KNN voting types when it comes to decide our next prediction
#uniform = each neighbour gets 1 vote regardless of distance, in case of equal votes, the first tracked neighbour wins
#distance = closer neighbours get more voting power
votingTypes = ["uniform", "distance"]

#trainSplits - Different proportions of data used for training
#0.2 = 20% training, 80% testing
#0.5 = 50% training, 50% testing
#0.8 = 80% training, 20% testing
trainSplits = [0.2, 0.5, 0.8]

#randomSeed - A fixed number to ensure reproducibility. This allows us to produce the same exact results every time
randomSeed = 42



#DATA PREPROCESSING FUNCTIONS

#SPLITTING DATASETS

#This function divides our data into two parts: training and testing. The model learns from the training data, then we test it on unseen test data. This tells us how well the model generalizes to new data
def splittingDatasets(X, Y, test_size=0.2, random_state=42):
    
    #X contains all the features (dependent variables)
    #Y contains all the labels (output/independent variable) that are options for us to choose our prediction on
    #test_size is the proportion of data to use for testing (default 20%)
    #random_state ensures we get the same split every time
    
    #train_test_split is from sklearn library
    #stratify=Y ensures each class appears in the same proportion in both sets
    #For example, if 30% of data is class A, both train and test will have ~30% class A
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=random_state,
        stratify=Y
    )
    
    return X_train, X_test, Y_train, Y_test


#SCALING FEATURES

#This function standardizes all features to have similar scales. Without scaling, features with large values would dominate the distance calculations
#For example: age (0-100) vs income (0-1000000) - income would matter much more
def scalingFeatures(X_train, X_test):
    
    #StandardScaler transforms data to have mean=0 and std=1. Formula: z = (x - mean) / std
    scaler = StandardScaler()
    
    #fit_transform on training data: calculates mean/std AND transforms. We only calculate mean/std from training data to avoid data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    
    #transform on test data: uses the SAME mean/std from training. This is important - we pretend we don't know anything about test data
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler



#HELPER FUNCTIONS

#METRICS CALCULATIONS

#This function computes the main metrics to evaluate our model performance.
def metricsCalculations(y_true, y_pred):
    
    #y_true = the actual correct labels
    #y_pred = what our model predicted
    
    #Accuracy: proportion of predictions that were correct
    #Example: 90 correct out of 100 = 0.90 accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    #Precision: when we predict a class, how often are we right?
    #High precision = few false positives
    #average="macro" means we calculate for each class then take the average
    #zero_division=0 prevents errors if a class has no predictions
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    
    #Recall: of all actual instances of a class, how many did we find?. High recall = few false negatives
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    #F1 Score: harmonic mean of precision and recall
    #Useful when you want a balance between precision and recall
    #Formula: 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return accuracy, precision, recall, f1


#MEMORY USAGE AMOUNT

#This function estimates how much memory our training data uses
#Helps us compare computational efficiency between implementations
def memoryUsageAmount(X_train, Y_train):
    
    #sys.getsizeof() returns the size of an object in bytes
    #We measure both the features (X) and labels (Y)
    memoryX = sys.getsizeof(X_train)
    memoryY = sys.getsizeof(Y_train)
    
    #Total memory is the sum of both
    return memoryX + memoryY


# CORE EXPERIMENT FUNCTION

#RUN MODEL

#This is the main experiment function
#It runs all combinations of settings for ONE dataset and ONE implementation
#For example: run all k values, all metrics, all voting types for Iris dataset with custom KNN
def runModel(dataset_name, implementation_type):
    
    #dataset_name = which dataset to use (e.g., "iris", "wine"). implementation_type = "custom" for our KNN or "sklearn" for the library version
    
    #This list will store all our experimental results
    finalResults = []
    
    #Load the dataset using our dataset_loader module
    #X = features, Y = labels
    #feature_names = names of each feature column
    #target_names = names of each class of outputs
    #info = dictionary with metadata (n_samples, n_features, n_classes)
    X, Y, feature_names, target_names, info = load_dataset(dataset_name)
    
    #Print some information so we can track progress
    print("Starting", info["name"], "-", implementation_type)
    
    #NESTED LOOPS - We test every combination of settings
    
    #Loop 1 - Different training set sizes
    for train_size in trainSplits:
        
        #Split the data into training and testing sets. test_size = 1 - train_size (e.g., if train is 0.8, test is 0.2)
        X_train, X_test, Y_train, Y_test = splittingDatasets(
            X,
            Y,
            test_size=1 - train_size,
            random_state=randomSeed
        )
        
        #Scale the features so they're all on similar scales. This is very important for distance-based methods like KNN
        X_train_scaled, X_test_scaled, scaler = scalingFeatures(
            X_train,
            X_test
        )
        
        #Measure how much memory the training data uses
        usedMemory = memoryUsageAmount(X_train_scaled, Y_train)
        
        #Loop 2: Different distance metrics
        for metric in distanceMetrics:
            
            #Loop 3: Different voting schemes
            for votingType in votingTypes:
                
                #Loop 4: Different k values
                for k in k_Range:
                    
                    #Safety check: skip if k is larger than our training set
                    #We can't find 100 neighbours if we only have 50 training points
                    if k > len(X_train):
                        continue
                    

                    #CUSTOM KNN IMPLEMENTATION
                    
                    if implementation_type == "custom":
                        
                        #Create our custom KNN model with current settings
                        model = customKNN_Model(
                            k=k,
                            metric=metric,
                            weights=votingType
                        )
                        
                        #Measure how long training takes. time.perf_counter() gives very precise time measurements
                        start_time = time.perf_counter()
                        model.fit(X_train_scaled, Y_train)
                        fit_time = time.perf_counter() - start_time
                        
                        #Measure how long prediction takes
                        start_time = time.perf_counter()
                        predictions = model.predict(X_test_scaled)
                        predict_time = time.perf_counter() - start_time
                    
                    # SKLEARN KNN IMPLEMENTATION
                    
                    else:
                        
                        #Create sklearn's KNN model with the same settings. Note: sklearn uses n_neighbors instead of k
                        model = KNeighborsClassifier(
                            n_neighbors=k,
                            metric=metric,
                            weights=votingType
                        )
                        
                        #Measure training time (same process as custom)
                        start_time = time.perf_counter()
                        model.fit(X_train_scaled, Y_train)
                        fit_time = time.perf_counter() - start_time
                        
                        #Measure prediction time
                        start_time = time.perf_counter()
                        predictions = model.predict(X_test_scaled)
                        predict_time = time.perf_counter() - start_time
                    
                    #Calculate how well the model performed
                    accuracy, precision, recall, f1 = metricsCalculations(
                        Y_test,
                        predictions
                    )
                    
                    #Store all results for this experiment. Each row contains all information about one experiment
                    finalResults.append([
                        dataset_name,
                        info["name"],
                        info["n_samples"],
                        info["n_features"],
                        info["n_classes"],
                        implementation_type,
                        k,
                        metric,
                        votingType,
                        train_size,
                        len(X_train),
                        len(X_test),
                        randomSeed,
                        accuracy,
                        precision,
                        recall,
                        f1,
                        fit_time,
                        predict_time,
                        usedMemory
                    ])
                    
                    #Print progress so we know the experiment is running
                    print(
                        "k =", k,
                        "| metric =", metric,
                        "| weights =", votingType,
                        "| train size =", train_size,
                        "| accuracy =", round(accuracy, 3)
                    )
    
    #Print summary when this dataset is done
    print("Finished dataset:", info["name"])
    print("Total experiments for this dataset:", len(finalResults))
    
    return finalResults


#RUNNING MODELS AND ALL DATASETS - since we have the function to run each type of model

#This function runs experiments on ALL datasets for BOTH implementations. It calls runModel() for each combination
def runningModels_and_Datasets():
    
    #Lists to store results separately for each implementation
    custom_results = []
    sklearn_results = []
    
    #Get list of all available datasets from our dataset_loader
    dataset_list = get_available_datasets()
    
    #Run experiments on each dataset
    for dataset_name in dataset_list:
        
        #First run with our custom KNN implementation
        results = runModel(
            dataset_name,
            "custom"
        )
        #extend() adds all items from results to custom_results
        custom_results.extend(results)
        
        #Then run with sklearn's KNN implementation
        results = runModel(
            dataset_name,
            "sklearn"
        )
        sklearn_results.extend(results)
    
    return custom_results, sklearn_results

#SAVING RESULTS TO CSV

#This function saves our results to a CSV file. CSV files can be easily loaded into R for statistical analysis

def csvProduction(filename, data):
    
    #filename = name of the file to create ("customKNN_results.csv"). data = list of rows to write
    
    #Column headers - these become the first row of the CSV
    #They describe what each column contains
    headers = [
        "dataset_id",
        "dataset_name",
        "n_samples",
        "n_features",
        "n_classes",
        "implementation",
        "k",
        "metric",
        "weights",
        "train_size_ratio",
        "train_size_actual",
        "test_size_actual",
        "seed",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "fit_time",
        "predict_time",
        "memory_bytes"
    ]
    
    #Open the file for writing
    #"w" means write mode (creates new file or overwrites existing)
    #newline="" prevents extra blank lines on Windows
    with open(filename, "w", newline="") as file:
        
        #csv.writer creates a writer object that can write rows
        writer = csv.writer(file)
        
        #First write the header row
        writer.writerow(headers)
        
        #Then write each data row
        for row in data:
            writer.writerow(row)
    
    print("\nResults successfully saved to:", filename)


#MAIN

#This is the main result production of our code. It decides the entire experiment process

def main():
    
    #Print a message so we know the script started
    print("Starting experiments...")
    
    #Run all experiments and get results for both implementations
    custom_results, sklearn_results = runningModels_and_Datasets()
    
    #Save results to separate CSV files
    #These files will be loaded into R for statistical analysis
    csvProduction("customKNN_results.csv", custom_results)
    csvProduction("sklearnKNN_results.csv", sklearn_results)
    
    #Print completion message
    print("Experiments finished.")


#In the end we call the main() function on __main__ so that it doesnt run everytime functions from this file are used
if __name__ == "__main__":
    main()
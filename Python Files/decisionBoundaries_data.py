#DECISION BOUNDARY DATA GENERATOR

#This file generates data for visualizing decision boundaries in R
#Decision boundaries show how the model divides the feature space into different classes. We create a grid of points and predict the class for each point
#This creates a "map" showing which areas belong to which class

#We use the Iris dataset with only 2 features so we can visualize in 2D. Using Petal Length and Petal Width because they separate the classes best


#IMPORTING LIBRARIES

#numpy - For creating arrays and grids of numbers
import numpy as np

#csv - For saving results to CSV files that R can read
import csv

#load_iris - To load the Iris dataset from sklearn
from sklearn.datasets import load_iris

#KNeighborsClassifier - sklearn's KNN for comparison
from sklearn.neighbors import KNeighborsClassifier

#Importing our custom KNN model
from customKNN_Model import customKNN_Model

#Importing the preprocessing functions from experiments.py
from experiments import splittingDatasets, scalingFeatures


#CONFIGURATION

#Random seed for reproducibility
randomSeed = 42

#Training size - we use 80% training, 20% testing
trainSize = 0.8

#k values to visualize - we pick 3 different values to show how k affects boundaries k=1 , k=5, k=20
k_Values = [1, 5, 20]

#We only use euclidean distance and uniform voting for visualization. Focused only on limited cases, just to showcase how decision boundaries look like and support some of our claims from our metrics.
distanceMetric = "euclidean"
votingType = "uniform"

#Which features to use from Iris dataset. Index 2 = Petal Length, Index 3 = Petal Width. These two features separate the three species best
feature1_Index = 2
feature2_Index = 3


#PREPARE DATA FUNCTION

#This function loads the Iris dataset and prepares it for boundary visualization
#We only use 2 features so we can plot in 2 dimensions

def prepareData():
    
    #Load the Iris dataset
    data = load_iris()
    
    #Get all features and labels
    X_full = data.data
    Y = data.target
    
    #Select only 2 features for 2D visualization
    #We use column indexing to pick Petal Length and Petal Width
    X = X_full[:, [feature1_Index, feature2_Index]]
    
    #Get the names of the features we're using
    feature1_name = data.feature_names[feature1_Index]
    feature2_name = data.feature_names[feature2_Index]
    
    print("Loading Iris dataset...")
    print("Using features:", feature1_name, "and", feature2_name)
    
    #Split into training and testing sets (same function as experiments.py)
    X_train, X_test, Y_train, Y_test = splittingDatasets(
        X,
        Y,
        test_size=1 - trainSize,
        random_state=randomSeed
    )
    
    #Scale the features (same function as experiments.py)
    X_train_scaled, X_test_scaled, scaler = scalingFeatures(
        X_train,
        X_test
    )
    
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    
    return X_train_scaled, Y_train, data


#CREATE GRID FUNCTION

#This function creates a grid of points covering the entire feature space
#We will predict the class for each point in this grid
#This creates the boundary colors in a decision boundary plot

def createGrid(X_train, resolution=100):
    
    #resolution controls how detailed the grid is
    #100 means 100x100 = 10,000 points
    
    #Find the minimum and maximum values for each feature
    #We add padding (0.5) so the grid extends slightly beyond the data
    x_min = X_train[:, 0].min() - 0.5
    x_max = X_train[:, 0].max() + 0.5
    y_min = X_train[:, 1].min() - 0.5
    y_max = X_train[:, 1].max() + 0.5
    
    #Create evenly spaced values for x and y axes
    #linspace creates 'resolution' evenly spaced numbers between min and max
    xx = np.linspace(x_min, x_max, resolution)
    yy = np.linspace(y_min, y_max, resolution)
    
    #meshgrid creates a 2D grid from the x and y values
    #Think of it like creating a checkerboard where we know every square's coordinates
    xx_grid, yy_grid = np.meshgrid(xx, yy)
    
    #Flatten the grid into a list of points
    #Each point is [x, y] coordinates
    #ravel() flattens the 2D array into 1D
    #np.c_ combines the x and y coordinates into pairs
    grid_points = np.c_[xx_grid.ravel(), yy_grid.ravel()]
    
    print("Grid created:", resolution, "x", resolution, "=", len(grid_points), "points")
    
    return grid_points


#GENERATE BOUNDARY DATA FUNCTION

#This function fits both KNN models and predicts classes for every point in the grid. We do this for each k value to see how k affects the boundaries

def generateBoundaryData(X_train, Y_train, grid_points):
    
    #List to store all results
    results = []
    
    #Loop through each k value
    for k in k_Values:
        
        #Skip if k is larger than training set
        if k > len(X_train):
            print("Skipping k =", k, "(larger than training set)")
            continue
        
        print("Processing k =", k, "...")
        
        #Create and train our custom KNN model
        custom_model = customKNN_Model(
            k=k,
            metric=distanceMetric,
            weights=votingType
        )
        custom_model.fit(X_train, Y_train)
        
        #Predict class for every point in the grid
        custom_predictions = custom_model.predict(grid_points)
        
        #Create and train sklearn's KNN model with same settings
        sklearn_model = KNeighborsClassifier(
            n_neighbors=k,
            metric=distanceMetric,
            weights=votingType
        )
        sklearn_model.fit(X_train, Y_train)
        
        #Predict class for every point in the grid
        sklearn_predictions = sklearn_model.predict(grid_points)
        
        #Store results for each grid point
        #Each row contains: x, y, custom prediction, sklearn prediction, k value
        for i in range(len(grid_points)):
            results.append([
                grid_points[i, 0],           #x coordinate
                grid_points[i, 1],           #y coordinate
                int(custom_predictions[i]),  #what custom KNN predicted
                int(sklearn_predictions[i]), #what sklearn KNN predicted
                k,                           #which k value was used
                distanceMetric,              #distance metric used
                votingType                   #voting type used
            ])
    
    return results


#GET TRAINING POINTS FUNCTION

#This function extracts the training points for plotting
#We need these to show the actual data points on top of the decision boundaries
def getTrainingPoints(X_train, Y_train):
    
    points = []
    
    #Loop through each training point
    for i in range(len(X_train)):
        points.append([
            X_train[i, 0],    #x coordinate
            X_train[i, 1],    #y coordinate
            int(Y_train[i])   #true class label
        ])
    
    return points


#SAVE BOUNDARY DATA FUNCTION

#This function saves the grid predictions to a CSV file
#R will load this file to create the decision boundary visualization
def saveBoundaryData(results, filename):
    
    #Column headers for the CSV file
    headers = [
        "x",
        "y",
        "custom_pred",
        "sklearn_pred",
        "k",
        "metric",
        "weights"
    ]
    
    #Open file and write data
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        for row in results:
            writer.writerow(row)
    
    print("Saved boundary data to:", filename)


#SAVE TRAINING POINTS FUNCTION

#This function saves the training points to a CSV file. R will use these to plot the actual data points on the visualization
def saveTrainingPoints(points, filename):
    
    headers = ["x", "y", "true_label"]
    
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        for row in points:
            writer.writerow(row)
    
    print("Saved training points to:", filename)


#MAIN FUNCTION

#This is the main function that runs the entire boundary data generation process
def main():
    
    print("Starting boundary data generation...")
    
    #Load and prepare the data
    X_train, Y_train, data = prepareData()
    
    #Create the grid of points
    grid_points = createGrid(X_train, resolution=100)
    
    #Generate predictions for the grid
    boundary_results = generateBoundaryData(X_train, Y_train, grid_points)
    
    #Get training points for plotting
    training_points = getTrainingPoints(X_train, Y_train)
    
    #Save results to CSV files
    saveBoundaryData(boundary_results, "boundary_data.csv")
    saveTrainingPoints(training_points, "training_points.csv")
    
    print("Boundary data generation finished.")


#In the end we call the main() function on __main__ so that it doesnt run everytime functions from this file are used
if __name__ == "__main__":
    main()
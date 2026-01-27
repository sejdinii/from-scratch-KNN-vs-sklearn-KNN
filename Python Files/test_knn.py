#TEST FILE FOR CUSTOM KNN MODEL

#This file contains unit tests for our custom KNN implementation to check if certain parts of the code work correctly
#We use pytest framework to run these tests (as advised)

#IMPORTING LIBRARIES

#pytest - A testing framework for Python. It finds and runs all functions that start with "test_"
import pytest
import sys
import os

#Add current directory to path so Python can find our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#Importing our custom KNN model that we want to test
from customKNN_Model import customKNN_Model


#DISTANCE METRIC TESTS
# ... rest of your code stays the same ...
#DISTANCE METRIC TESTS

#These tests check if our distance calculations are correct. We use known mathematical values to verify the results

def test_euclidean_distance_known_values():
    #Testing euclidean distance with a 3-4-5 right triangle
    #This is a famous Pythagorean triple where 3^2 + 4^2 = 5^2
    #So the distance from (0,0) to (3,4) should be exactly 5
    
    #Create a KNN model (we just need it to access the distance function)
    knn = customKNN_Model(k=3, metric='euclidean')
    
    #Define two points
    x1 = [0.0, 0.0]  #Origin point
    x2 = [3.0, 4.0]  #Point at (3,4)
    
    #Calculate the distance using our function
    result = knn.euclideanDistance(x1, x2)
    
    #The expected answer is 5 (from Pythagorean theorem)
    expected = 5.0
    
    #Check if our result matches the expected value
    assert result == expected


def test_manhattan_distance_known_values():
    #Testing manhattan distance
    #Manhattan distance is just |x1-x2| + |y1-y2|
    #From (0,0) to (3,4): |0-3| + |0-4| = 3 + 4 = 7
    
    #Create a KNN model
    knn = customKNN_Model(k=3, metric='manhattan')
    
    #Define two points
    x1 = [0.0, 0.0]
    x2 = [3.0, 4.0]
    
    #Calculate the distance using our function
    result = knn.manhattanDistance(x1, x2)
    
    #The expected answer is 7
    expected = 7.0
    
    #Check if our result matches
    assert result == expected

#FIT AND PREDICT TESTS

#These tests check if the basic fit and predict functions work
def test_fit_stores_data():
    #The fit() function should store the training data in the model
    #After calling fit(), X_train and Y_train should not be None
    
    knn = customKNN_Model(k=3)
    
    #Create some training data
    X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    Y_train = [0, 1, 0]
    
    #Fit the model
    knn.fit(X_train, Y_train)
    
    #Check that data was stored
    assert knn.X_train is not None
    assert knn.Y_train is not None
    assert len(knn.X_train) == 3


def test_predict_returns_correct_length():
    #predict() should return one prediction for each test point
    #If we give it 2 test points, we should get 2 predictions back
    
    knn = customKNN_Model(k=1)
    
    X_train = [[0.0, 0.0], [10.0, 10.0]]
    Y_train = [0, 1]
    knn.fit(X_train, Y_train)
    
    #Predict for 2 test points
    X_test = [[0.1, 0.1], [9.9, 9.9]]
    predictions = knn.predict(X_test)
    
    #Should get 2 predictions back
    assert len(predictions) == 2


#VOTING SCHEME TESTS

#These tests check if our voting mechanisms work correctly
def test_uniform_voting():
    #In uniform voting, each neighbour gets exactly 1 vote
    #The class with the most votes wins
    
    knn = customKNN_Model(k=3, weights='uniform')
    
    #Create training data where classes are clearly separated
    #Points near origin are mostly class 0
    X_train = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [10.0, 10.0]]
    Y_train = [0, 0, 1, 1]
    knn.fit(X_train, Y_train)
    
    #Test point very close to origin
    #Nearest 3 neighbours are at (0,0), (0.1,0.1), (0.2,0.2) with labels [0, 0, 1]
    #Majority vote: class 0 has 2 votes, class 1 has 1 vote
    #So prediction should be class 0
    predictions = knn.predict([[0.05, 0.05]])
    
    assert predictions[0] == 0


def test_distance_weighted_voting():
    #In distance-weighted voting, closer neighbours get more influence
    #Weight = 1/distance, so very close neighbours dominate
    
    knn = customKNN_Model(k=3, weights='distance')
    
    #Create training data
    #Point at (0,0) is class 1
    #Points at (1,1) and (1.1,1.1) are class 0
    X_train = [[0.0, 0.0], [1.0, 1.0], [1.1, 1.1]]
    Y_train = [1, 0, 0]
    knn.fit(X_train, Y_train)
    
    #Test point very close to (0,0)
    #Even though 2 neighbours are class 0, the nearest one (class 1) is much closer
    #Because of distance weighting, class 1 should win
    predictions = knn.predict([[0.01, 0.01]])
    
    assert predictions[0] == 1


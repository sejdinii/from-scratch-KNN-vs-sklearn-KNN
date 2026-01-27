
#CUSTOM KNN MODEL

#This file contains our own implementation of the K-Nearest Neighbours
#The following is the KNN model implementation completely from scratch without the use of any existing KNN implementation libraries


class customKNN_Model:
    
    #INITIALIZATION

    #Below are the functions that we use to build our KNN model
    #The implementation is built on fitting, predicting, accepting k neighbour values and operating on two distance metrics and two voting schemes
    def __init__(self, k=3, metric='euclidean', weights='uniform'):
        
        #k is the number of neighbours we look at when making a prediction
        #For example, if k=3, we look at the 3 closest training points
        #Default is 3, but we can change it when creating the model
        
        #metric is which distance formula we use to measure "closeness"
        #Options are 'euclidean' or 'manhattan'
        #Default is 'euclidean'
        
        #weights is how we count the votes from neighbours
        #'uniform' means each neighbour gets 1 vote
        #'distance' means closer neighbours get more voting power
        #Default is 'uniform'
        
        #First we check if k is valid - it must be a positive number
        #We cannot look at 0 or negative number of neighbours
        if k <= 0:
            raise ValueError("k must be positive")
        
        #We store all the settings as attributes of the model
        #self.k, self.metric, self.weights can be accessed later
        self.k = k
        self.metric = metric
        self.weights = weights
        
        #These will store our training data later
        #For now they are empty
        self.X_train = None  #This holds the feature values
        self.Y_train = None  #This holds the output class labels
    
    

    #FIT FUNCTION (TRAINING Process of the Model)

    #In KNN, fit/training is very simple - we just memorize the data.
    #Unlike other algorithms, KNN does not learn any patterns. It simply stores all training examples and uses them later.
    def fit(self, X_train, Y_train):
        
        #X_train contains the features (measurements) of our training data
        #For example, in the Iris dataset: petal length, petal width, etc.
        
        #Y_train contains the output variable (what class each sample belongs to)
        #For example: 0 = setosa, 1 = versicolor, 2 = virginica

        #Error handling!
        #We check that X_train and Y_train have the same number of samples
        #Each feature row must have exactly one corresponding label
        if len(X_train) != len(Y_train):
            raise ValueError("X_train and Y_train must have same length")
        

        #We store the training data in the model
        #This data will be used later when we make predictions
        self.X_train = X_train
        self.Y_train = Y_train
    
    
    #EUCLIDEAN DISTANCE

    #This function calculates the straight-line distance between two points.
    #It uses the Pythagorean theorem extended to multiple dimensions. Formula: sqrt( (x1-x2)^2 + (y1-y2)^2 + ... )
    def euclideanDistance(self, x1, x2):
        
        #x1 and x2 are two data points 
        #For example: x1 = [1.5, 2.0, 3.1] and x2 = [1.0, 2.5, 3.0]
        
        #We start with distance = 0 and add up squared differences
        distance = 0.0
        
        #We loop through each feature (dimension)
        #range(len(x1)) gives us indices: 0, 1, 2, ... for each feature
        for i in range(len(x1)):
            
            #For each feature, we calculate (x1[i] - x2[i])^2
            #This is the squared difference for this dimension
            distance += (x1[i] - x2[i]) ** 2
        
        #Finally we take the square root of the sum
        return distance ** 0.5
    
    

    #MANHATTAN DISTANCE

    #This function calculates distance as if walking on a grid. Formula: |x1-x2| + |y1-y2| + ...
    def manhattanDistance(self, x1, x2):
        
        #x1 and x2 are two data points (lists of numbers)
        
        #We again start with distance = 0 and add up absolute differences
        distance = 0.0
        
        #We loop through each feature (dimension)
        for i in range(len(x1)):
            
            #For each feature, we calculate |x1[i] - x2[i]|
            #abs() gives us the absolute value (always positive)
            distance += abs(x1[i] - x2[i])
        
        return distance
    
    

    #DISTANCE TYPE SELECTOR
 
    #This function chooses which distance formula to use based on the metric setting we specified when creating the model.
    def distanceType(self, x1, x2):
        
        #We check which metric was chosen in the model implementation code and call the appropriate function
        if self.metric == 'euclidean':
            #Use Euclidean distance
            return self.euclideanDistance(x1, x2)
        
        elif self.metric == 'manhattan':
            #Use Manhattan distance
            return self.manhattanDistance(x1, x2)
        
        else:
            #If an unknown metric was specified, default to Euclidean
            #Just a function to be safe for any wrong inputs
            return self.euclideanDistance(x1, x2)
    
    

    # PREDICT FUNCTION

    #his is the main function that makes predictions for new data and showcases the true logic of KNN.
    #For each new point, it calculates distance to ALL training points, finds the k closest neighbours, uses voting to decide the class
    
    def predict(self, X_test):
        
        #X_test contains the new data points we want to classify. It can be one point or many points
        
        #Error Handling!
        #First we check that the model has been trained
        #If fit() was never called, X_train and Y_train will be None
        if self.X_train is None or self.Y_train is None:
            raise RuntimeError("You must train the model before predicting")
        
        #Error Handling!
        #We check that k is not larger than our training set
        #We cannot find 100 neighbours if the training split has only 50 training points
        if self.k > len(self.X_train):
            raise ValueError(f"k={self.k} is larger than training set size={len(self.X_train)}")
        
        #This list will store our predictions for each test point
        predictions = []
        
        #We loop through each test point that needs a prediction
        for test_point in X_test:
            
 
            #Step 1 - Calculate distance to all training points
            
            #This list will store (distance, label) pairs
            distances = []
            
            #enumerate() gives us both the index (i) and the value (train_point)
            #i goes from 0 to number of training points
            for i, train_point in enumerate(self.X_train):
                
                #Calculate distance from test_point to this training point
                dist = self.distanceType(test_point, train_point)
                
                #Get the class label(output) for this training point
                label = self.Y_train[i]
                
                #Store both the distance and the label together
                distances.append((dist, label))
            
            #Step 2 - Find the k nearest neighbours
            
            #We sort the list by distance (smallest first)
            #lambda x: x[0] means "sort by the first element" referring to the distance
            distances.sort(key=lambda x: x[0])
            
            #We take only the first k entries - the k closest neighbours
            k_nearest = distances[:self.k]
            
            #Step 3: Voting to decide the class
            
            #We create an empty dictionary to count votes for each class
            #Keys will be class labels, values will be vote counts
            vote_counts = {}
            
            if self.weights == 'uniform':
                
                #UNIFORM VOTING 
                #Each neighbour gets exactly 1 vote. The class with the most votes wins
                
                #We loop through each of our k nearest neighbours
                for dist, label in k_nearest:
                    
                    #Check if we have seen this label before
                    if label in vote_counts:
                        #If yes, add 1 to its count
                        vote_counts[label] += 1
                    else:
                        #If no, start counting at 1
                        vote_counts[label] = 1
            
            else:
                
                # DISTANCE WEIGHTED VOTING
                #Closer neighbours get more votes. Weight = 1 / distance (closer = smaller distance = bigger weight)
                
                for dist, label in k_nearest:
                    
                    #Calculate weight as 1/distance
                    #We add 0.0001 to avoid division by zero if distance is 0
                    weight = 1.0 / (dist + 0.0001)
                    
                    #Check if we have seen this label before
                    if label in vote_counts:
                        #If yes, add the weight to its count
                        vote_counts[label] += weight
                    else:
                        #If no, start counting at this weight
                        vote_counts[label] = weight
            
            #Step 4: Get the winning class / final prediction
            
            #We need to find which class has the highest vote count. We start by assuming no winner yet
            best_label = None
            best_count = -1
            
            #We loop through all classes and their vote counts
            #vote_counts.items() gives us pairs like (label, count)
            for label, count in vote_counts.items():
                
                #If this class has more votes than our current best
                if count > best_count:
                    #Update the best to be this class
                    best_count = count
                    best_label = label
            
            #best_label now contains the class with the most votes
            predicted_label = best_label
            
            #Add this prediction to our results list
            predictions.append(predicted_label)
        
        #Returning all predictions
        return predictions
    
    
    #GETING PARAMETERS

    #This function returns the current settings of the model. Useful for checking what configuration was used.
    def get_parameters(self):
        
        #We return a dictionary with all the model settings
        return {
            'k': self.k,
            'metric': self.metric,
            'weights': self.weights
        }
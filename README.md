- Author: Inderjit Singh Sidhu
- Profile: https://github.com/indrsidhu
- Code hosted at: https://github.com/indrsidhu/machine_learning_study_increments

# Import libs
```py
import numpy as np
import pandas as pd
```

# sigmoid function standarize inputs, it make sure your value is between 0 and 1
```py
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
```

# GRADED FUNCTION: initialize_with_zeros
```py
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b
```

# propgate function take w,b as input and try to 
# reduce cost (difference between actual output and predicted output)
```py
def propogate(w,b,X,Y):
    m = X.shape[0]
    A = sigmoid(np.dot(w.T,X)+b)
    #cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))   # compute cost
    cost = (1/m) * np.sum((np.subtract(A,Y))**2)
    
    cost = np.squeeze(cost)
    
    difference = (A-Y)
    dw = (1 / m) * np.dot(X, difference.T)
    db = (1 / m) * np.sum(difference)
    return [dw,db,cost]
```    

# optimizer is responsible for finding best optimal weight, part of gradient decent
```py
def optimizer(w,b,X,Y, num_iterations, learning_rate):
    for i in range(num_iterations):
        dw, db, cost = propogate(w,b,X,Y)
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db )
    return [w,b,cost]
```

# This predict function is final function which perform w.T*x+b where we provide 
# weight, bias and input , it calculate prediction for each x value in input array
# w,b are constant once these are optimized by optimizer
```py
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if(A[0][i]<=0.5):
            Y_prediction[0][i] = A[0][i] #0
        else:
            Y_prediction[0][i] = A[0][i] #1
    
    return Y_prediction
```

# This is model which gives us optimized weight and bias value by reducing cost
```py
def model(w,b,X,Y,num_iterations,learning_rate):
    w,b,cost = optimizer(w,b,X,Y,num_iterations,learning_rate)

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction = predict(w, b, X)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))    
    print("train cost: {}".format(cost))    
    
    return [w,b]
```

# === START TRAINING WITH EXISTING ==
# Study HR departments data of employees increments
```py
df = pd.read_csv('data.csv')
X = df["Increment"] # Employees increment in RS
Y = df["Happy"] # Employees satisfaction for given increment where 1='happy', 0='unhappy'

#convert input and output to (nx,m) shape where nx=each inputs data length which is 1 in our case
# m= number of inputs which is 20 in oour case
# make X and Y of (1,20) shape by transposing array function
X = X.reshape(X.shape[0], -1).T
Y = Y.reshape(Y.shape[0], -1).T

# Standarizing is very important, to keep inputs between 0 and 1 range
# so divide input X with its maximum increment which willfalldata between 0-1 range for all inputs
X = X / np.amax(X)

# initialize weight and bias to 0 in starting,our model will find optimized 
# weight and bias values for training data automatically
w, b = initialize_with_zeros(X.shape[0])

num_iterations = 10000 # training data will be itrated number of times to find best optimized w and b values
learning_rate = 0.0004 # while learning slope will be changed with this given rate
w,b = model(w,b,X,Y,num_iterations,learning_rate) # run this model to start traning
```


# once we have w and b value, now for any future prediction we can save w and b and can apply to any future data
# ==== FUTER PREDICTION BASED ON MODEL ===

# For new year we are planning to give this increment to our 3 employees
# check with past experiance (model we trained), if this increment will make employees happy or not

```py
predictInput_orig = np.array([2000,10000,15000])

# standardize input for shape like (nx,m)
predictInput_orig = predictInput_orig.reshape(predictInput_orig.shape[0],-1).T
# stanardize each input (fall it under 0 to 1 range),save this in new variable 
# so that we print original salary at end of output
predictInput = predictInput_orig / np.amax(predictInput_orig)

# w,b are constant we find after traning, for any futer data w,b will be constant
prediction = predict(w,b,predictInput)
# print each input's prediction seperately with user friendly format
for i in range(predictInput.shape[1]):
    print("RS {} increment to employees salary, make them {} % Happy".format(predictInput_orig[0][i],int(prediction[0][i]*100)))
```

# OUTPUT WILL LOOK LIKE THIS
- RS 2000 increment to employees salary, make them 25 % Happy
- RS 10000 increment to employees salary, make them 81 % Happy
- RS 15000 increment to employees salary, make them 95 % Happy
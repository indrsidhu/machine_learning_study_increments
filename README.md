This small machine learning program explains how you can train model to predict future outputs.
Problem: A company’s is planning to give increment to their three employees, based on increments study (past experience) try to predict if given amount will make employees happy or not.
By running py script we calculate w (weight) and b (bias) values which will helps to solve your problem.

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



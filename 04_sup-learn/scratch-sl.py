# Scratch Python

# import the linear_model from sklearn
from sklearn import linear_model

# create and train a regression
reg = linear_model.LinearRegression()
reg = reg.fit(train_features, train_outcomes)

# extracting info from sklearn
reg.predict([list of value(s)]) # predicting on new data
reg.coef_ # returns the coefficients
reg.intercept_ # returns the intercept_

reg.score(test_features, test_outcomes) # r-squared score on test
reg.score(train_features, train_outcomes) # r-squared score on train
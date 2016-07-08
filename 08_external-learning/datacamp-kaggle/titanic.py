import pandas as pd

### Loading in Data ###

# load the data as DataFrame objects with pandas read_csv
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


### Explore the data ###

# .describe() method summarizes columns/features of the DataFrame (like summary function in R)
train.describe()

# .shape gives you the dimensions of the data set
train.shape

# we can run value_counts() on a column to see how many survived or died
train["Survived"].value_counts()
# set normalize = True in value_counts to get the output as percent instead of value_counts
train["Survived"].value_counts(normalize=True)

# We can furthur filter the data to look at survival counts for men or women
train["Survived"][train["Sex"]=="male"].value_counts(normalize=True)
# result: 81% of men died, 18% survived
train["Survived"][train["Sex"]=="female"].value_counts(normalize=True)
# result: 26% of women died, 74% survived

# Adding new columns in pandas
# we can add a new column simply with your_data["new_var"] = 0 (this will initialize all values of "new_var" to 0)
# create a new variable called "Child", returns 1 if Age < 18
train["Child"] = float('NaN')

train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0

# now we can look at survival rates for children
train["Survived"][train["Child"]==1].value_counts(normalize=True)
train["Survived"][train["Child"]==0].value_counts(normalize=True)
# 53% of children survived, while 38% of adults survived


### Cleaning and Formatting Data ###

# from train.describe(), we can see that Age has some missing values. We would like to impute these values.
# we use the .fillna method to set the missing values to median Age
train["Age"] = train["Age"].fillna(train["Age"].median())

# we will also impute the Embarked field, choosing the most common location, which is "S"
train["Embarked"] = train["Embarked"].fillna("S")

# now we will change some categorical variables into numeric categories (male=0, female=1; embarked S=0, C=1, Q=2)
train["Sex"][train["Sex"]=="male"] = 0
train["Sex"][train["Sex"]=="female"] = 1

train["Embarked"][train["Embarked"]=="S"] = 0
train["Embarked"][train["Embarked"]=="C"] = 1
train["Embarked"][train["Embarked"]=="Q"] = 2


### Predictions ###

## A Basic Prediction: All Men Must Die

# For our first prediction, we will assume that all females survive and all males die
test_one = test # create a copy of the test set (remember, this doesn't have a Survived field)
test_one["Survived"] = 0 # initialize Survived to 0
test_one["Survived"][test_one["Sex"]=="female"] = 1

## Decision Tree 1

# import the necessary packages: numpy and sklearn's tree
import numpy as np
from sklearn import tree

# this creates an array called target of the outcomes
target = train["Survived"].values
# this creates an array of the features we want to use
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit a decision tree
my_tree_one = tree.DecisionTreeClassifier() # initialize classifier object
my_tree_one = my_tree_one.fit(features_one, target) # fit on the features

# See most important classifiers, how model performed:
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))

# Predict and submit to Kaggle

# first, there is a missing value in Fare in the test set
test.Fare[152] = test["Fare"].median()
# note: I tried filtering test where Fare == NaN, did not work for me

# we also have to clena the test features like we did for train
test["Age"] = test["Age"].fillna(test["Age"].median())

test["Embarked"] = test["Embarked"].fillna("S")

test["Sex"][test["Sex"]=="male"] = 0
test["Sex"][test["Sex"]=="female"] = 1

test["Embarked"][test["Embarked"]=="S"] = 0
test["Embarked"][test["Embarked"]=="C"] = 1
test["Embarked"][test["Embarked"]=="Q"] = 2

# create our array of test features
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# create prediction set using test set
my_prediction = my_tree_one.predict(test_features)

# Now we need to create a data frame with two columns: PassengerId and Survived. Survived = predictions.
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])

# We write our solution to csv with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label=["PassengerId"])

## Decision Tree 2 - controlling for overfitting

# In first decision tree, we used default arguments for max_depth and min_samples_split (which are None).
# Note that this model does worse than our simple gender model. This is probably due to overfitting
my_tree_one.score(features_one, target) # accuracy on training set of 97.76 %
# accuracy on test set of 71.29 %

# let's create another model controlling for some overfitting
features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]] # we add a few features

# set max_depth, min_samples_split
max_depth = 10
min_samples_split = 5 
# initialize and train model
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

# How does it perform on training data?
print(my_tree_two.score(features_two, target))
# 90.57 %, worse than model 1 performed on the training data.

test_features_two = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction_two = my_tree_two.predict(test_features_two)
my_solution_two = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])

my_solution_two.to_csv("my_solution_two.csv", index_label=["PassengerId"])
# score on testing data: 71.29 % (this actually has the exact same outcome as model 1, just lower accuracy on training)

## Decision Tree 3 - feature engineering

# create new training set with an extra feature, family size = SibSp + Parch + 1
train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
# do the same for test set
test_two = test.copy()
test_two["family_size"] = test_two["SibSp"] + test_two["Parch"] + 1

# create features
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
test_features_three = test_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# create new classifier
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# How does model perform on training data (notice that we didn't set the max depth and min sample split)
print(my_tree_three.score(features_three, target))
# 97.98 %


## Predicting with Random Forest

# first import the new classifier
from sklearn.ensemble import RandomForestClassifier
# note: we will have to set the n_estimators parameter when using this classifer

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(features_forest, target)

# How does random forest model score on the training set?
print(my_forest.score(features_forest, target))
# 93.94 %

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

# Predict on testing data, submit to Kaggle
my_prediction_forest = my_forest.predict(test_features)
my_solution_forest = pd.DataFrame(my_prediction_forest, PassengerId, columns=["Survived"])
my_solution_forest.to_csv("my_solution_forest.csv", index_label=["PassengerId"])
# Accuracy on testing data: 75.12 %, an improvement over the decision tree







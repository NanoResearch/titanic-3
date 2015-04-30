""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

"""
import pandas as pd
import numpy as np
import csv as csv
import pylab as P
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
## Adapted from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split

#REGRESSION FOR SETTING MISSING VALUES
def setMissingAges(df):
    # Grab all the features that can be included in a Random Forest Regressor
    age_df = df[['Age','Embarked', 'Parch', 'SibSp','Pclass']]
    # Split into sets with known and unknown Age values
    knownAge = age_df.loc[ (df.Age.notnull()) ]
    unknownAge = age_df.loc[ (df.Age.isnull()) ]
    # All age values are stored in a target array
    y = knownAge.values[:, 0]
    # All the other values are stored in the feature array
    X = knownAge.values[:, 1::]
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(unknownAge.values[:, 1::])
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age'] = predictedAges
    return df

# Standard Scaler will subtract the mean from each value then scale to the unit variance
scaler = preprocessing.StandardScaler()

# Data cleanup
###################################### TRAINING DATA ####################################################
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a data frame

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values # dropna() removes NaN

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

#setMissingAges(train_df)

#FEATURE SCALING FOR TRAINING SET
train_df['Fare'] = scaler.fit_transform(train_df['Fare'])

#lin----------------
train_df['SibSp*Gender'] = train_df['SibSp']*train_df['Gender']
train_df['SibSp*Fare'] = train_df['SibSp']*train_df['Fare']
train_df['SibSp*Age'] = train_df['SibSp']*train_df['Age']
train_df['SibSp*Pclass'] = train_df['SibSp']*train_df['Pclass']
train_df['SibSp^2*Pclass'] = train_df['SibSp']*train_df['SibSp']*train_df['Pclass']
train_df['SibSp^2*Age'] = train_df['SibSp']*train_df['SibSp']*train_df['Age']
train_df['Gender*Fare'] = train_df['Gender']*train_df['Fare']
train_df['Age*Fare'] = train_df['Age']*train_df['Fare']
train_df['Port*Gender'] = train_df['Embarked']*train_df['Gender']
train_df['Port*SibSp'] = train_df['Embarked']*train_df['SibSp']
train_df['Port*Pclass'] = train_df['Embarked']*train_df['Pclass']
train_df['Port*Fare'] = train_df['Embarked']*train_df['Fare']
train_df['Port*SibSp*Gender'] = train_df['Embarked']*train_df['SibSp']*train_df['Gender']

###################################### TEST DATA ####################################################
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
#train_df = train_df.rename(columns = {'Age_scaled':'Age'}, inplace=True)
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a data frame

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

#setMissingAges(test_df)

#FEATURE SCALING FOR TEST SET
#test_df['Fare'] = scaler.fit_transform(test_df['Fare'])

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[test_df.Pclass == f+1]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1), 'Fare'] = median_fare[f]

#lin--------------------
test_df['SibSp*Gender'] = test_df['SibSp']*test_df['Gender']
test_df['SibSp*Fare'] = test_df['SibSp']*test_df['Fare']
test_df['SibSp*Age'] = test_df['SibSp']*test_df['Age']
test_df['SibSp*Pclass'] = test_df['SibSp']*test_df['Pclass']
test_df['SibSp^2*Pclass'] = test_df['SibSp']*test_df['SibSp']*test_df['Pclass']
test_df['SibSp^2*Age'] = test_df['SibSp']*test_df['SibSp']*test_df['Age']
test_df['Gender*Fare'] = test_df['Gender']*test_df['Fare']
test_df['Age*Fare'] = test_df['Age']*test_df['Fare']
test_df['Port*Gender'] = test_df['Embarked']*test_df['Gender']
test_df['Port*SibSp'] = test_df['Embarked']*test_df['SibSp']
test_df['Port*Pclass'] = test_df['Embarked']*test_df['Pclass']
test_df['Port*Fare'] = test_df['Embarked']*test_df['Fare']
test_df['Port*SibSp*Gender'] = test_df['Embarked']*test_df['SibSp']*test_df['Gender']
#----------------------

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::, 1::], train_data[0::, 0])

print 'Predicting...'
output = forest.predict(test_data).astype(int)

predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

#train_df.hist()
P.show()

# assume classifier and training data is prepared...
features_list = train_df.columns.values[1::]
X = train_df.values[:, 1::]
y = train_df.values[:, 0]

train_sizes, train_scores, test_scores = learning_curve(
        forest, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(.1, 1., 10), verbose=0)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("RandomForestClassifier")
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.ylim((0.6, 1.01))
plt.gca().invert_yaxis()
plt.grid()

# Plot the average training and test score lines at each training set size
plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")

# Plot the std deviation as a transparent range at each training set size
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                 alpha=0.1, color="b")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                 alpha=0.1, color="r")

# Draw the plot and reset the y-axis
plt.draw()
plt.show()
plt.gca().invert_yaxis()

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
forest.fit(X_train, y_train)

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(y_test, forest.predict_proba(X_test)[:,1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)
print 'ROC AUC: %0.2f' % roc_auc

# Plot of a ROC curve for a specific class
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
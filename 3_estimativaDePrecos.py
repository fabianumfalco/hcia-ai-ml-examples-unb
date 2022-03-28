# Ignore errors not to polute the presentation
import warnings

warnings.filterwarnings("ignore")
# Introduce the basic package of data science.
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
# Introduce machine learning, preprocessing, model selection, and evaluation indicators.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
# Import the Boston dataset used this time.
from sklearn.datasets import load_boston
# Introduce algorithms.
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, ElasticNet
# Compared with SVC, it is the regression form of SVM.
from sklearn.svm import SVR
# Integrate algorithms.
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import math
import random
import numpy

# Load the Boston house price data set.
boston = load_boston()
# x features, and y labels.
x = boston.data
y = boston.target
# Display related attributes.
print('Feature column name')
print(boston.feature_names)
print("Sample data volume: %d, number of features: %d" % x.shape)
print("Target sample data volume: %d" % y.shape[0])

x = pd.DataFrame(boston.data, columns=boston.feature_names)
x.head()

fig, ax = plt.subplots(ncols=1)
ax.hist(tuple(y), density=True, bins=20)
pd.DataFrame(y).plot(kind='density', ax=ax)
ax.set_xlabel("Average Price of Houses (x USD 1,000)")

# Segment the data.
# 80% goes into the training set
# 20% goes into the testing set
# random_state is a shuffling seed (fixed for reproducible results)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)
print("Before:")
print(x_train[0:10])

# Store original dataset values before standardizing the data
x_train_og = x_train
x_test_og = x_test

# Standardize the data set.
# StandardScaler removes the mean and scales to unit variance.
# z = (x - u) / s
ss = StandardScaler()
x_train = ss.fit_transform(x_train)  # Fit the data and transform to scale the training set
x_test = ss.transform(x_test)  # Performs standardization on the test set
# WARNING:
# the ss.transform() standardization use the mean values and variance from the training set from ss.fit_transform()
# if the mean and variance of the test set is dissimilar to the training set, bad things can happen
print("After:")
print(x_train[0:10])

# Set the regression model name and instantiate the model
regression_models = {
    'LinerRegression': LinearRegression(),
    'Ridge': RidgeCV(alphas=(0.001, 0.1, 1), cv=3),
    'Lasso': LassoCV(alphas=(0.001, 0.1, 1), cv=5),
    'Random Forrest': RandomForestRegressor(n_estimators=10),
    'GBDT': GradientBoostingRegressor(n_estimators=30),
    'Support Vector Regression': SVR(),
    'ElasticNet': ElasticNet(alpha=0.001, max_iter=10000),
    'XgBoost': XGBRegressor()
}


# cv is the cross-validation idea here.
# Output the R2 scores of all regression models.
# Define the R2 scoring function.
def R2(model, x_train, x_test, y_train, y_test):
    model_fitted = model.fit(x_train, y_train)
    y_pred = model_fitted.predict(x_test)
    score = r2_score(y_test, y_pred)
    # print("Predito:\n", y_pred, "\nMedido:\n", y_test)
    return score


# Traverse all models to score.
for (name, model) in regression_models.items():
    score = R2(model, x_train, x_test, y_train, y_test)
    print("{}: {:.6f}".format(name, score.mean()))

x_test_example = x_train_og[:10].loc[x_train_og[:10]["ZN"] > 0]
x_train_example = x_train_og[:10].loc[x_train_og[:10]["ZN"] < 1]
print("Before standardization")
print("xtraining\n", x_train_example, "\n\n", "xtesting\n", x_test_example)

y_test_example = [y_train[:10][4], y_train[:10][8]]
y_train_example = [*y_train[:4], *y_train[5:8], *y_train[9:10]]
# print(y_train[:10])
# print(y_train_example)
# print(y_test_example)
ss1 = StandardScaler()
x_train_example = ss1.fit_transform(x_train_example)  # Fit the data and transform to scale the training set
x_test_example = ss1.transform(x_test_example)  # Performs standardization on the test set
# print("\n\nAfter standardization")
# print("xtraining\n", x_train_example, "\n\n", "xtesting\n", x_test_example)

print("\n\n")
# Traverse all models to score with the wrong dataset.
for (name, model) in regression_models.items():
    score = R2(model, x_train_example, x_test_example, y_train_example, y_test_example)
    print("{}: {:.6f}".format(name, score.mean()))

# Traverse all models to score with the correct dataset.
for (name, model) in regression_models.items():
    score = R2(model, x_train, x_test, y_train, y_test)
    print("{}: {:.6f}".format(name, score.mean()))

plt.figure(figsize=(16, 8), facecolor='w')
##Perform visualization.
ln_x_test = range(len(x_test))

# Draw known prices in the test set
plt.plot(ln_x_test, y_test, lw=4, label=u'Real prices in the test set')

# Set legend, grid, plot title and limit x-axis range
plt.grid(True)
plt.title(u"Boston Housing Price Forecast")
plt.xlim(0, 101)

# Plot lines for each model prediction
for (name, model) in regression_models.items():
    y_predict = model.predict(x_test)
    plt.plot(ln_x_test, y_predict, lw=2,
             label=u'Predicted prices with %s, $R^2$=%.3f' % (name, r2_score(y_test, model.predict(x_test))))
plt.legend(loc='upper left')

plt.show()

parameters = {
    'kernel': ['linear', 'rbf', 'poly'],  # kernel function
    'C': [0.1, 0.5, 0.9, 1, 5],  # SVR regularization factor
    'gamma': [0.001, 0.01, 0.1, 1]
    # 'rbf', 'poly' and 'sigmoid' kernel function coefficient, which affects the model performance
}
# Use grid search and perform cross validation.
model = GridSearchCV(SVR(), param_grid=parameters, cv=3)
model.fit(x_train, y_train)

# Instantiate SVR with different kernels
svrs = {"Linear": SVR(kernel="linear"),
        "Polynomial": SVR(kernel="poly"),
        "RBF": SVR(kernel="rbf"),
        }


# Create a dataset with 100 samples of a given function and split into test and train sets
def dataset(func, samples=100):
    x_train = list(range(samples))
    y_train = list(map(lambda x: func(x), x_train))
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=28)
    return x_train, x_test, y_train, y_test


# Create 3 datasets with 2*x+noise, x^2-x*noise+noise, sin(x/10)
datasets = {
    "Linear": dataset(lambda x: x * 2 - random.gauss(0, 2)),
    "Polynomial": dataset(lambda x: x ** 2 - x * random.gauss(0, 2) + random.gauss(0, 3)),
    "Sine": dataset(lambda x: math.sin(x / 10)),
}


def test_svr_kernels(datasets):
    # for each kernel and dataset, train the model, predict and plot the results in a grid
    fig, axis = plt.subplots(nrows=3, sharex=True, figsize=(10, 15))

    # Set legend, grid, plot title and limit x-axis range
    plt.grid(True)
    plt.title("Test different SVR kernels with different datasets")

    i = 0
    for (datasettype, dataset) in datasets.items():
        x_train = dataset[0]
        y_train = dataset[2]
        x_test = dataset[1]
        y_test = dataset[3]
        axis[i].scatter(*list(zip(*sorted(zip(x_test, y_test)))), lw=2, label=u'Expected values')
        axis[i].set_title("Dataset %s" % datasettype)
        # Plot lines for each model prediction
        for (svrkernel, svr) in svrs.items():
            svr.fit(numpy.array([x_train]).reshape(-1, 1), numpy.array([y_train]).reshape(-1, 1))
            y_predict = svr.predict(numpy.array([x_test]).reshape(-1, 1))
            score = r2_score(y_test, y_predict)

            # *list(zip(*sorted(zip(x_test,y_test))))
            # is a trick to merge the x and y lists into a single lists with (x,y) pairs,
            # then sort them, and finally split back into separate lists of x and y
            # If values passed to axis.plot or matplotlib.pyplot.plot are not sorted, incorrect lines will be drawn
            axis[i].plot(*list(zip(*sorted(zip(x_test, y_predict)))), lw=2,
                         label=u'Predicted with %s kernel, $R^2$=%.3f' % (svrkernel, score))
            axis[i].legend(loc='lower right')
        # Change to the next image
        i += 1

    plt.show()


# Show visually how different kernels fit to different data
test_svr_kernels(datasets)

print("Optimal parameter list:", model.best_params_)
print("Optimal model:", model.best_estimator_)
print("Optimal R2 value:", model.best_score_)
print("R2 score with the test set:", r2_score(y_test, model.predict(x_test)))

##Perform visualization with the model predicting target y values based on the test dataset
ln_x_test = range(len(x_test))
y_predict = model.predict(x_test)
# Set the canvas.
plt.figure(figsize=(16, 8), facecolor='w')
# Draw with a red solid line.
plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'Value')
# Draw with a green solid line.
plt.plot(ln_x_test, y_predict, 'g-', lw=3, label=u'Estimated value of the SVR algorithm, $R^2$=%.3f' %
                                                 r2_score(y_test, model.predict(x_test)))
# Display in a diagram.
plt.legend(loc='upper left')
plt.grid(True)
plt.title(u"Boston Housing Price Forecast (SVM)")
plt.xlim(0, 101)
plt.show()

##Perform visualization with the model predicting target y values based on the training dataset
ln_x_train = range(len(x_train))
y_predict = model.predict(x_train)
# Set the canvas.
plt.figure(figsize=(16, 8), facecolor='w')
# Draw with a red solid line.
plt.plot(ln_x_train, y_train, 'r-', lw=2, label=u'Value')
# Draw with a green solid line.
plt.plot(ln_x_train, y_predict, 'g-', lw=3, label=u'Estimated value of the SVR algorithm, $R^2$=%.3f' %
                                                  r2_score(y_train, model.predict(x_train)))
# Display in a diagram.
plt.legend(loc='upper left')
plt.grid(True)
plt.title(u"Boston Housing Price Forecast (SVM)")
plt.xlim(0, 101)
plt.show()

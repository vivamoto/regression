# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:11:34 2020

@author: Victor Ivamoto


Disciplina de Aprendizado de Máquina

Atividade No 1 - Classificador Linear

Nesta atividade o aluno deve implementar um classificador linear
para problema binário e com múltiplas classes.

Os conjuntos de dados (diabetes, hepatitis, iris) a serem tratados
estão disponíveis no site do UCI Machine Learning
(https://archive.ics.uci.edu/ml/index.php).

O aluno deve baixar os conjuntos de dados e realizar o preprocessamento dos
dados (codificação, dados faltantes, normalização) antes de projetar o
classificador linear. Caso o conjunto de dados não esteja particionado em
trenamento e test, o aluno deve utilizar 2/3 do conjunto de dados para
treinamento e 1/3 para test. Neste caso, deve-se escolher para
o conjunto de treinamento a parte inicial dos dados.

O aluno deve postar no site do tidia o código e um relatório. No relatório
deve constar a taxa de classficação correta para o conjunto de dados. No caso
de regularização, deve ser mencionado como foi escolhido o parâmetro de
regularização.

Data da entrega: 20/04/2020

====================================
Origem dos datasets:

hepatitis:
https://archive.ics.uci.edu/ml/datasets/Hepatitis

Iris:
https://archive.ics.uci.edu/ml/datasets/Iris

Diabetes
https://www.kaggle.com/uciml/pima-indians-diabetes-database
"""
#######################
# 0. Preparação
#######################

# Inicia o kernel com o comando abaixo
# python -m spyder_kernels.console

# Altera a pasta de trabalho
cd "D:\\Documentos\\Profissão e Carreira\\Mestrado\\Aprendizado de Máquina\\Atividade_1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special

#######################
# 1. Dataset preparation
#######################
# This section downloads the dataset, check missing values,
# and normalizes the data.

import requests
import os.path

#======================
# 1.1 Download datasets
#======================

# This function downloads the datasets from UCI site
# Since the diabetes dataset is in Kaggle, we are unable
# to download automatically. Access the site, download the file
# and save in this code working directory.
def getDataset(url, folder, fname):

  # Create folder to save the files
  if not os.path.exists(folder):
    os.mkdir(folder)

  # Download file and store in variable "myfile"
  myfile = requests.get(url)

  # Save file in disk
  open(folder + "/" + fname, 'wb').write(myfile.content)

# Download hepatitis dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
getDataset(url = url, folder = "hepatitis", fname = "hepatitis.data")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.names"
getDataset(url = url, folder = "hepatitis", fname = "hepatitis.names")

# Download iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
getDataset(url = url, folder = "iris", fname = "iris.data")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names"
getDataset(url = url, folder = "iris", fname = "iris.names")

# Load the datasets
iris     = pd.read_csv("iris/iris.data", names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

hepatitis = pd.read_csv("hepatitis/hepatitis.data", names = ["Class", "AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY", ])

diab_train = pd.read_csv("diabetes/dataset_train.txt", sep = '\t')
diab_test = pd.read_csv("diabetes/dataset_test.txt", sep = '\t')


# Inspect the imported datasets
diab_train.head()
diab_test.head()
hepatitis.head()
iris.head()

#======================
# 1.2 Outcome: Multi-class classification - 1 of n
#======================
# We create a column for each class and use the 1 of n approach
#       X.w = Y
# Where:
#   X is the matrix of features,
#   w is the weight vector and
#   Y is the outcome vector.
#
# In this activity, only the iris dataset is multi class.
# Iris: create a column for each flower class, using 1 of n
# For example, the 'setosa' column contains 1 for setosa observations and 0 otherwise
iris.insert(loc = iris.shape[1], column = "setosa", value = (iris["class"] == "Iris-setosa").astype('int') , allow_duplicates = True)

iris.insert(loc = iris.shape[1], column = "versicolor", value = (iris["class"] == "Iris-versicolor").astype('int') , allow_duplicates = True)

iris.insert(loc = iris.shape[1], column = "virginica", value = (iris["class"] == "Iris-virginica").astype('int') , allow_duplicates = True)

iris.head()
iris.dtypes

#======================
# 1.3 Missing values
#======================
# The hepatitis dataset contains many missing values, represented by "?"
# Since these "?" are spread in the dataset, we'll remove them carefully.

# Before deleting any row or column, let's check the matrix size, so
# we can compare later.
hepatitis.shape

# The first approach is to remove the rows and columns with the maximum
# number of "?".

# Drop rows with more than 20% missing values
# We store the row numbers in the "row" variable, and later we drop
# these rows.
i = 0       # counter to loop over all rows
row = []    # Stores the row numbers to delete
while i < hepatitis.shape[0]:
    # Find row with more than 20% of "?"
    if np.mean(hepatitis.iloc[i,:] == "?") > 0.2:
        row.append(i)
    i = i + 1
# Delete the rows
hepatitis.drop(row, inplace = True)

# Drop columns with more than 20% missing values
# We do the same as above, but for columns.
# We save the column numbers in "col" variable, and later
# we delete these columns.
i = 0
col = []    # stores column numbers
while i < hepatitis.shape[1]:
    # Find columns to delete
    if np.mean(hepatitis.iloc[:,i] == "?") > 0.2:
        col.append(i)
    i = i + 1
# Delete columns
hepatitis.drop(columns = hepatitis.columns[col], inplace = True)

# The numeric columns, we replace the missing values with the mean
# value of the feature.
# Loop over all dataframe columns
for i in range(14, 18):

    # Calculate the column mean
    new_value = hepatitis[hepatitis.iloc[:,i] != "?"].iloc[:,i].astype(float).mean()

    # Replace the "?" value with the new value
    hepatitis.iloc[:,i].replace(to_replace = "?", value = new_value, inplace = True)

    # Convert the column to float
    hepatitis.iloc[:,i] = hepatitis.iloc[:,i].astype(float)


# Convert columns to real numbers
# Return the columns not converted
def real_num():
    col = []
    for i in range(1, hepatitis.shape[1]):
        try:
            hepatitis.iloc[:, i] = hepatitis.iloc[:, i].astype(float)
        except:
            col.append(i)
            print('Column ' + str(i) + ' contains "?"')
    return col

col = real_num()
# Count the number of "?" remaining in each column
np.sum(hepatitis.iloc[:,col] == "?")
col
hepatitis.loc[hepatitis.iloc[:,3] == "?", :].iloc[:,3:10]
hepatitis.loc[hepatitis.iloc[:,8] == "?", :].iloc[:,3:10]
hepatitis.loc[hepatitis.iloc[:,9] == "?", :].iloc[:,3:10]
hepatitis.dtypes

# There are 6 rows with missing values
row = hepatitis.loc[hepatitis.iloc[:,3] == "?", :].index
hepatitis.drop(row, inplace = True)

row = hepatitis.loc[hepatitis.iloc[:,9] == "?", :].index
hepatitis.drop(row, inplace = True)

# Convert columns to real numbers
real_num()

hepatitis.shape

# Inspect data types
hepatitis.dtypes

#======================
# 1.4 Categorical variables: -1 and 1
#======================
# Regression uses the signal function to calculate the outcome
# So, we convert all categorical values to -1 and 1

# Diabetes
diab_train['class'].replace(to_replace = 0, value = -1, inplace = True)
diab_test['class'].replace(to_replace = 0, value = -1, inplace = True)

# Hepatitis: convert categorical variables with 1 and 2 to 0 and 1.
for i in ['Class', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']:
    hepatitis[i] = hepatitis[i] - 1
    hepatitis[i].replace(to_replace = 0, value = -1, inplace = True)

# Iris
for i in ['setosa', 'versicolor', 'virginica']:
    iris[i].replace(to_replace = 0, value = -1, inplace = True)

#======================
# 1.5 Create the training and test sets
#======================
# The train and test dasasets for diabetes were provided
# We use 2/3 of the data set for training and 1/3 for test.
# For hepatitis, the initial part is used for training.

# Iris
# Use every 3 rows for test set and the other 2 for training.
# Example:
# Train rows = 0, 1, 3, 4, 6, 7, ...
# Test rows  = 2, 5, 8, 11, ...
row = range(iris.shape[0])  # Range of row numbers, start with 0
row = np.array(row) + 1     # Convert to vector and add 1
iris_train = iris.iloc[np.mod(row, 3) != 0]  # Skip every 3 rows
iris_test  = iris.iloc[np.mod(row, 3) == 0]  #

# Hepatitis
# The initial part is used for training.
mid = int(hepatitis.shape[0]*2/3)
end  = hepatitis.shape[0]
hep_train = hepatitis[0:mid]   # Training
hep_test  = hepatitis[mid:end] # Test


#======================
# 1.6 Data normalization
#======================
# Each dependent variable is normalizaded using min-max or z-score.
# min-max transforms the values to fit in the range [0, 1].
# The formula is:
# x = (x - xmin) / (xmax - xmin),
# where 'xmax' is the maximum value and 'xmin' is the minimum value
# z-score uses normal distribution with mean = 0 and standard deviation = 1
def norm_z(X, mu = 0, sd = 0):
    if mu == 0 or sd == 0:
        mu = X.mean()
        sd = np.std(X)
    return (X - mu) / sd, mu, sd

def norm_minmax(X, min = 0, max = 0):
    if min == 0 or max == 0:
        xmin = X.min()
        xmax = X.max()

    return (X - xmin) / (xmax - xmin), xmin, xmax

# Normalize diabetes
for i in range(diab_test.shape[1] - 1) :
    diab_train.iloc[:,i], mu, sd = norm_z(X = diab_train.iloc[:,i])
    diab_test.iloc[:,i],  mu, sd = norm_z(X = diab_test.iloc[:,i], mu = mu, sd = sd)

# Normalize hepatitis
# Continuous dependent variables
for i in ['AGE', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN']:
    hep_train.loc[:,i], mu, sd = norm_z(X = hep_train[i])
    hep_test.loc[:,i],  mu, sd = norm_z(X = hep_test[i], mu = mu, sd = sd)

# Normalize iris
for i in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    iris_train.loc[:,i], mu, sd = norm_z(X = iris_train[i])
    iris_test.loc[:,i],  mu, sd = norm_z(X = iris_test[i], mu = mu, sd = sd)


#######################
# 3. Function definitions
#######################

#==================================
# 3.1 Linear regression error function
#==================================
# Error function for linear regression
def error(X, y, w, l, q = 1):
    # Input values:
    # X: matrix of coefficients
    # y: vector of outputs
    # w: vector of weights
    # l: lambda (scalar)
    # q: 1 for lasso, 2 for ridge regression
    # Output: error

    # Formula in slide 43
    # Least squares
    y = np.array(y)
    E = 0.
    N = X.shape[0]  # Number of observations (rows in X matrix)
    d = X.shape[1]  # Number of columns in X matrix
    for i in range(N):
        s = 0.
        for j in range(d):
            s = s + X[i,j] * w[j]
        E += (s - y[i]) ** 2

    # Calculate regularization term
    M = w.shape[0]
    reg = 0.
    for j in range(M):
        reg = reg + abs(w[j]) ** q

    # Total error
    E = E / N + l * reg

    # Return the error
    return np.mean(E)

#==================================
# 3.2 Linear Regression - Regularization
# Return weights with regularization
#==================================
# This function calculates the regularization coefficient lambda
# We find lambda that maximizes the accuracy.
# So, we split the train set in train and validation sets.
# Compute the weights and lambda in train and use the validation set
# to calculate the accuracy.
def calc_reg(X, y, X_val, y_val, k):
    # Input parameters:
    # X: matrix of features coefficients in train set
    # y: vector of outcomes in train set.
    # X_val: X in validation set
    # y_val: y in validation set
    # k: Number of fold in use (k-fold cross validation)

    step = 0.1      # lambda increase step
    N = X.shape[1]  # Number of attributes + bias
    I = np.eye(N)   # Identity matrix

    # Calculate initial weights
    # Rormula in lesson 2, slide 31:
    # w = (X_T * X)^-1 * X_T * y
    if np.linalg.det(X.T @ X) == 0.:
        w = np.linalg.pinv(X.T @ X) @ X.T @ y
    else:
        w = np.linalg.inv(X.T @ X) @ X.T @ y

    # Calculate the initial error (Emin)
    # Formula in slide 42
    wmin = w    # Initial value of w
    lmin = 0    # Initial value of lambda
    Emin = error(X = X_val, y = y_val, w = wmin, l = lmin * step)

    # This dataframe used to create a chart
    df = pd.DataFrame(data = {'Counter' : k, 'Lambda' : [0], 'Error': Emin, 'Mean Weight' : w.mean()})

    # Test several values of lambda(l) and pick the lambda
    # that miminizes the error (slide 41)
    for l in range(100):
        # Weight with regularization (slide 43)
        if np.linalg.det(((l * step) / N) * I + X.T @ X) == 0.:
            w = np.linalg.pinv(((l * step) / N) * I + X.T @ X) @ X.T @ y
        else:
            w = np.linalg.inv( ((l * step) / N) * I + X.T @ X) @ X.T @ y

        # Calculate the error with the new weights and lambda
        E = error(X = X_val, y = y_val, w = w, l = l * step)

        # Pick the values of minimum error
        if E <= Emin:
            wmin = w        # Weights of mimimum error
            lmin = l * step # Lambda of mimimum error
            Emin = E        # Minimum error

        # Create a dataframe to plot the values
        #df = df.append({'Lambda' : l, 'Accuracy' : acc}, ignore_index=True)
        df = df.append({'Counter' : k, 'Lambda' : l * step, 'Error' : E, 'Mean Weight' : w.mean()}, ignore_index=True)

    # Return the best values: weight, lambda, accuracy and data frame
    return df, Emin, wmin, lmin

#==================================
# 3.3 Linear Regression - Regularization
# k-fold cross validation for regularization
#==================================
# The objective is estimate the parameter lambda that
# results in the lowest error in the validation set.
# We use k-fold cross validation for this: we split
# the training set into training and validation sets
# with 90% and 10% of the original training set size.
def regularization(X, y, K = 10):
    # Input parameters:
    # X: trainning set matrix with predictors (features)
    # y: trainning set vector with outcomes

    # Split the train set in train and validation sets
    # 80% for train and 20% for validation
    N = X.shape[0]
    mid = int( N / K)

    # Train set
    X_train = X[0:mid]
    y_train = y[0:mid]

    # Validation set
    X_val = X[mid:N]
    y_val = y[mid:N]

    df, Emin, wmin, lmin = calc_reg(X = X_train, y = y_train, X_val = X_val, y_val = y_val, k = 0)

    # k-fold cross validation, where k = 10.
    for k in range(K):

        # Define 10% of rows for validation set
        N = X.shape[0]
        rv = np.array(range(int(N * k / K),
                            int(N * (k + 1) / K)))

        # Define complementary row numbers for train set
        r = np.setdiff1d(np.array(range(X.shape[0])), rv)

        # Create the train set
        X_train = X[r]
        y_train = y[r]

        # Create the validation set
        X_val = X[rv]
        y_val = y[rv]

        # Compute the weights with regularization factor
        df1, E, w, l = calc_reg(X = X_train, y = y_train, X_val = X_val, y_val = y_val, k = k+1)

        if k == 0:
            df = df1
        else:
            df = df.append(df1)

        # Minimum error
        if E < Emin:
            Emin = E    # Error
            wmin = w    # Weight
            lmin = l    # lambda

    # Calculate the best weight with regularizaiton
#    w, l, acc, df = calc_reg(X_train, y_train, X_val, y_val, multi_class)
#    wmax, lmax, acc_max, df, Emin, wmin, lmin = calc_reg(X_train, y_train, X_val, y_val, multi_class)

#    return w, l, acc, df
#    return wmax, lmax, acc_max, df, Emin, wmin, lmin
    return df, Emin, wmin, lmin

#==================================
# 3.4 Calculate the derivatives
#==================================
# Return the sigmoid and softmax derivatives
# used in logistic regression
def calc_derivative(X, y, w, binary = True):
    # Input definition
    # x: vector of feature coeficients
    # y: vector of outcomes
    # w: vector of weights
    # binary: True for binary classification (sigmoid),
    #         False for multi-class (softmax)
    # Output: Gradient of sigmoid or softmax
    if binary:
        # Calculate the sigmoid gradient
        N = X.shape[0]
        # Aula 2, slide 68
        s = 0.
        for n in range(N):
            s = s + (y[n] * X[n]) / (1 + np.exp(y[n] * w @ X[n]))
        g = -1/N * s
    else:
        # Calculate softmax gradient
        # Used in multiclass logistic regression
        # Formula in page 35, definition 4.20
        # CS480/680–Fall 2018 - University of Waterloo

        # p: probability of y being 1
        p = scipy.special.softmax(w.T @ X.T, axis = 1)

        # Calculate the gradient
        g = X.T @ (p - y.T).T

        N = X.shape[0]
        g = 0.
        for i in range(N):
            x_i = np.array([X[i]]).T
            y_i = np.array([y[i]]).T
            #p_i = scipy.special.softmax(w.T @ x_i, axis = 1)
            p_i = scipy.special.softmax(w.T @ x_i)

            # Calculate the gradient
            g = g + x_i @ (p_i - y_i).T

    return g


#==================================
# Variable Learning rate (lr)
#==================================
# Compute the variable learning rate of gradient
# descent used in binary logistic regression
def calc_lr(X, y, w, d, binary = True):
    # d = direction
    # binary: True for binary classification (sigmoid),
    #         False for multi-class (softmax)
    np.random.seed(1234)
    epsilon = 1e-3
    hlmin = 1e-3
    lr_l = 0                # Lower lr
    lr_u = np.random.rand() # Upper lr

    # New w position
    wn = w + lr_u * d

    # Calculate the gradient of new position
    g = calc_derivative(X = X, y = y, w = wn, binary = binary)

    hl = g.T @ d

    while hl < 0 :
        #
        lr_u = 2 * lr_u

        # Calculate the new position
        wn = w + lr_u * d

        # Calculate the gradient of new position
        # f and h aren't used
        g = calc_derivative(X = X, y = y, w = wn, binary = binary)

        hl = g.T @ d

    # lr medium is the average of lrs
    lr_m = (lr_l + lr_u) / 2

    # Estimate the maximum number of iterations
    itmax = np.ceil(np.log ((lr_u - lr_l) / epsilon))

    # Iteration counter
    it = 0

    while abs(hl) > hlmin and it < itmax :

       # Calculate new position
        wn = w + lr_m * d

        # Calculate the gradient of the new position
        # Note: f and h aren't used
        g = calc_derivative(X = X, y = y, w = wn, binary = binary)

        hl = g.T @ d
        if hl > 0 :
            # Decrease upper lr
            lr_u = lr_m
        elif hl < 0 :
            # Increase lower lr
            lr_l = lr_m
        else:
            break

        # lr medium is the lr average
        lr_m = (lr_l + lr_u) / 2

        # Increase number of iterations
        it = it + 1

    return lr_m


#==================================
# Logistic: Error function
#==================================
# Logistic regression error function
def error_lr(w, X, y, binary):
    if binary:
        # Formula in slide 56
        E = 0.
        N = y.shape[0]  # Size of X
        for n in range(N):
            E = E + np.log(1 + np.exp(-y[n] * w @ X[n]))
        E = E / N
    else:
        c = y.shape[0]
        n = y.shape[1]
        E = 0.
        for i in range(n):
            for k in range(c):
                x_i = np.array([X[i]]).T
                f = scipy.special.softmax(w.T @ x_i, axis = 0)
                E = E - y[k,i] * np.log10(f)

        E = sum(np.sum(y @ np.log10(scipy.special.softmax(w.T @ X.T, axis = 0) + 1e-300), axis =1))
    return E

#==================================
# Theta is the sigmoid function used in
# binary logistic regression
#==================================
def theta(s):
    # Slide 46
    theta = np.exp(s) / (1 + np.exp(s))
    return theta

#==================================
# Gradient Descent for Logistic Regression
#==================================
# Returns the gradient for both binary and
# multi-class logistic regression.
# Fixed and variable learning rates (lr) available
# Implementation of algorithm in slide 79
#+++++++++++++++++++++++++++++++++
def gradient(X, y, v = True, binary = True, maxiter = 1000, lr=0.1):
    # Input parameters
    # X: matrix of coefficients
    # y: vector of binary outcomes
    # v: True for variable learning rate (lr)
    # binary: True for binary classification (sigmoid),
    #         False for multi-class (softmax)
    # maxiter: maximum number of iterations
    # lr: learning rate.
    # Output: weights vector
    normagrad = 1e-10     # Maximum gradient norm

    np.random.seed(123)
    # Step 1: Initial weights with random numbers
    if binary:
        w = np.random.rand(X.shape[1]) * 100

    else:
        w = np.random.rand(X.shape[1], y.shape[1]) * 100

    # Calculate initial gradient (g)
    g = calc_derivative(X = X, y = y, w = w, binary = binary)

    # Step 2: For t = 0, 1, 2, ... do
    t = 0
    while t < maxiter and np.linalg.norm(g) > normagrad:

        # Increase the number of iterations
        t = t + 1

        # Step 3: Calculate the new gradient
        g = calc_derivative(X = X, y = y, w = w, binary = binary)

        # Step 4: Calculate learning rate (lr) for variable
        # gradient descent and binary classification
        if v and binary:
            lr = calc_lr(X = X, y = y, w = w, d = -g, binary = binary)

        # Step 5: Update weight (w)
        w = w - lr * g

    # Return the weights vector, gradient and error
    return w, g

#==================================
# Plot results
#==================================
def plot(df, yval, title):
    # Input:
    # df = dataframe with values to be plotted
    # yval = column name in df of y-axis variable
    # title = chart title
    for i in df['Counter'].unique():
        x = df.Lambda[df.Counter == i]
        y = df[yval][df.Counter == i]
        label = 'k = ' + str(i)
        plt.plot(x, y, label = label)  # Plot some data on the (implicit) axes.
    plt.xlabel('Lambda')
    plt.ylabel(yval)
    plt.title(title)
    plt.legend()

#######################
# 4. Diabetes dataset
#######################
# Independent variable (y):
y_train = diab_train.iloc[:,diab_train.shape[1]-1]

y_test = diab_test.iloc[:,diab_test.shape[1]-1]
y_test = np.array(y_test)

# Dependent variables (X):
X_train = diab_train.iloc[:,0:diab_train.shape[1]-1]
X_train = np.array(X_train)
X_train = np.insert(X_train, 0, 1, axis = 1)    # Add column x0 = 1 for the bias

X_test = diab_test.iloc[:,0:diab_test.shape[1]-1]
X_test = np.array(X_test)
X_test = np.insert(X_test, 0, 1, axis = 1)

#----------------------
# 4.1 Diabetes: Linear regression
#----------------------
# Compute the weights (w) using the formula in lesson 2, slide 31:
# w = (X_T * X)^-1 * X_T * y
w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Now that we have the weights (w), we'll apply them in
# the test set and compare with real values.
# Prediction in the test set
y_hat = np.sign(X_test @ w)

# Compute the accuracy
acc = (100 * np.mean(y_hat == y_test)).round(2)
acc

results = pd.DataFrame(data = {'Dataset' : 'Diabetes', 'Method' : 'Regession', 'Accuracy' : [acc]})
results

#----------------------
# 4.2 Diabetes: Linear regression with Regularization
#----------------------
# Calculate the weights with regularization
#w, l, acc, df,  = regularization(X_train, y_train)
df, E, w, l  = regularization(X = X_train, y = y_train, K = 10)
y_hat = np.sign(X_test @ w)

# Compute the accuracy
acc = (100 * np.mean(y_hat == y_test)).round(2)

results = results.append({'Dataset' : 'Diabetes',
                          'Method' : 'Reg. w/ regularization',
                          'Accuracy' : acc},
                          ignore_index = True)

results
# Plot Lambda x Error
plot(df = df, yval = 'Error', title = 'Diabetes - Regularization')


#----------------------
# 4.3 Diabetes: Logistic regression
#----------------------
# Variable gradient descent, maximum number of iteration  = 1k, learning rate = 1
w, g = gradient(X = X_train, y = y_train, v = True, binary = True, maxiter = 1000, lr = 1)

# Return 1 if P(y=1|X=x) > 50%, return 0 otherwise
y_hat = (theta(s = X_test @ w) >= 0.5) * 1

# Convert 0 to -1
y_hat = np.where(y_hat == 0, -1 , y_hat)

# Compute the accuracy
acc = (100 * np.mean(y_hat == y_test)).round(2)

# Gradient size
print(np.linalg.norm(g))

results = results.append({'Dataset' : 'Diabetes',
                          'Method' : 'Logistic',
                          'Accuracy' : acc},
                          ignore_index = True)

results
#######################
# 5. Hepatitis Dataset
#######################
# O dataset de hepatitis tem a coluna "Class" com dois valores, 0 e 1
# "DIE" e "LIVE". Esse dataset pode ser usado para classificação  binária.

# Independent variable (Y):
y_train = hep_train['Class']
y_train = np.array(y_train)

y_test = hep_test['Class']
y_test = np.array(y_test)

# Dependentes variable (X):
X_train = hep_train.iloc[:,1:hep_train.shape[1]]  # Demais colunas
X_train = np.array(X_train)
X_train = np.insert(X_train, 0, 1, axis = 1)

X_test = hep_test.iloc[:,1:hep_test.shape[1]]  # Demais colunas
X_test = np.array(X_test)
X_test = np.insert(X_test, 0, 1, axis = 1)

#----------------------
# 5.1 Hepatitis: Linear regression
#----------------------
# Compute the weights (w) using the formula in lesson 2, slide 31:
# w = (X_t*X)^-1 *X_t * y
w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

y_hat = np.sign(X_test @ w)

# Compute the accuracy
acc = (100 * np.mean(y_hat == y_test)).round(2)

results = results.append({'Dataset' : 'Hepatitis',
                          'Method' : 'Regession',
                          'Accuracy' : acc},
                          ignore_index = True)

results

#----------------------
# 5.2 Hepatitis: Linear regression with Regularization
#----------------------
# Calculate the weights with regularization
df, E, w, l  = regularization(X = X_train, y = y_train, K = 10)
y_hat = np.sign(X_test @ w)

# Compute the accuracy
acc = (100 * np.mean(y_hat == y_test)).round(2)

results = results.append({'Dataset' : 'Hepatitis',
                          'Method' : 'Reg. w/ regularization',
                          'Accuracy' : acc},
                          ignore_index = True)

results
# Plot Lambda x Error
plot(df = df, yval = 'Error', title = 'Hepatitis - Regularization')

#----------------------
# 5.3 Hepatitis: Logistic regression
#----------------------
# fixed gradient descent, maximum number of iteration  = 1k, learning rate = 1
w, g = gradient(X = X_train, y = y_train, v = False, binary = True, maxiter = 1000, lr = 1)

# Return 1 if P(y=1|X=x) > 50%, return 0 otherwise
y_hat = (theta(s = X_test @ w) >= 0.5) * 1

# Convert 0 to -1
y_hat = np.where(y_hat == 0, -1 , y_hat)

# Compute the accuracy
acc = (100 * np.mean(y_hat == y_test)).round(2)
acc
print(np.linalg.norm(g))

results = results.append({'Dataset' : 'Hepatitis',
                          'Method' : 'Logistic',
                          'Accuracy' : acc},
                          ignore_index = True)

results
#######################
# 6. Iris Dataset
#######################
# Iris dataset contains 3 classes and is used for multiclass classification

# Independent variable (Y):
y_train = iris_train.loc[:,['setosa', 'versicolor', 'virginica']]

y_test = iris_test.loc[:,['setosa', 'versicolor', 'virginica']]
y_test = np.array(y_test)

# Dependent variables (X):
X_train = iris_train.loc[:,['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
X_train = np.array(X_train)
X_train = np.insert(X_train, 0, 1, axis = 1)

X_test = iris_test.loc[:,['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
X_test = np.array(X_test)
X_test = np.insert(X_test, 0, 1, axis = 1)

#----------------------
# 6.1 Iris: Linear regression
#----------------------
# Calculate the weights using formula in lesson 2, slide 31:
# w = (X_t*X)^-1 *X_t * y
w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predict new values with the optimized weights and
# X values from the test set
y_hat = X_test @ w
for i in range(y_hat.shape[0]):
    y_hat.iloc[i] = (y_hat.iloc[i] == y_hat.iloc[i].max())* 1
y_hat.replace(0, -1, inplace = True)

# Compute the accuracy
acc = (100 * np.mean(pd.DataFrame(y_hat) == y_test)).round(2)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Reg. - setosa ',
                          'Accuracy' : acc[0]},
                          ignore_index = True)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Reg. - versicolor ',
                          'Accuracy' : acc[1]},
                          ignore_index = True)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Reg. - virginica ',
                          'Accuracy' : acc[2]},
                          ignore_index = True)

# Overall accuracy
acc = (100 * np.mean(np.array(y_hat) == y_test)).round(2)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Reg. - overall',
                          'Accuracy' : acc},
                          ignore_index = True)

results
#----------------------
# 6.2 Iris: Linear regression with regularization
#----------------------
# Calculate the weights with regularization
y_train = np.array(y_train)
df, Emin, wmin, lmin  = regularization(X = X_train, y = y_train, K = 10)
y_hat = X_test @ wmin
for i in range(y_hat.shape[0]):
    y_hat[i] = y_hat[i] == np.max(y_hat[i])
y_hat = np.where(y_hat == 0, -1 , y_hat)

# Compute the accuracy
acc = (100 * np.mean(pd.DataFrame(y_hat) == y_test)).round(2)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Reg. w/ regulariz - setosa ',
                          'Accuracy' : acc[0]},
                          ignore_index = True)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Reg. w/ regulariz - versicolor ',
                          'Accuracy' : acc[1]},
                          ignore_index = True)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Reg. w/ regulariz - virginica ',
                          'Accuracy' : acc[2]},
                          ignore_index = True)

# Overall accuracy
acc = (100 * np.mean(np.array(y_hat) == y_test)).round(2)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Reg. w/ Regulariz. - Overall',
                          'Accuracy' : acc},
                          ignore_index = True)

results
# Plot Lambda x Error
plot(df = df, yval = 'Error', title = 'Iris - Regularization')

#----------------------
# 6.3. Iris: Logistic Regression
#----------------------
# y_train and y_test values are {-1, 1}, and for
# logistic regression we need {0, 1}.
# So, these lines make the convertion.
y_train = np.where(y_train == -1, 0, y_train)
y_test  = np.where(y_test  == -1, 0, y_test)

# Calculate the weights
# fixed gradient descent, maximum number of iteration  = 10k, learning rate = 2
w, g = gradient(X = X_train, y = y_train, v = False, binary = False, maxiter = 10000, lr = 2)

# Make y_hat prediction with softmax
y_hat = scipy.special.softmax(w.T @ X_test.T, axis = 0)
# Get the maximum value and transpose
y_hat = ((y_hat == np.max(y_hat, axis = 0)) * 1).T
y_hat.shape

# Compute the accuracy
acc = (100 * np.mean(pd.DataFrame(y_hat) == y_test)).round(2)
acc
print(np.linalg.norm(g))

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Logistic - setosa ',
                          'Accuracy' : acc[0]},
                          ignore_index = True)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Logistic - versicolor ',
                          'Accuracy' : acc[1]},
                          ignore_index = True)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Logistic - virginica ',
                          'Accuracy' : acc[2]},
                          ignore_index = True)

# Overall accuracy
acc = (100 * np.mean(np.array(y_hat) == y_test)).round(2)

results = results.append({'Dataset' : 'Iris',
                          'Method' : 'Logistic - Overall',
                          'Accuracy' : acc},
                          ignore_index = True)

results


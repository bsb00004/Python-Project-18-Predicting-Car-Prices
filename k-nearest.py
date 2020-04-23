#!/usr/bin/env python
# coding: utf-8

# ## Python Project 18: Predicting Car Prices
# In this project, we'll use the machine learning workflow (using K-nearest neighbour technique) to predict a car's market price using its attributes. The data set we will be working with contains information on various cars. For each car we have information about the technical aspects of the vehicle such as the motor's displacement, the weight of the car, the miles per gallon, how fast the car accelerates, and more. You can read more about the data set [here](https://archive.ics.uci.edu/ml/datasets/automobile). Here's a documentation of the data set:
# 
# | Attribute         | Attribute Range                                                                                                                                                                                |
# |-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | symboling         | -3, -2, -1, 0, 1, 2, 3.                                                                                                                                                                        |
# | normalized-losses | continuous from 65 to 256.                                                                                                                                                                     |
# | make              | alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz, mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota, volkswagen, volvo |
# | fuel-type         | diesel, gas.                                                                                                                                                                                   |
# | aspiration        | std, turbo.                                                                                                                                                                                    |
# | num-of-doors      | four, two.                                                                                                                                                                                     |
# | body-style        | hardtop, wagon, sedan, hatchback, convertible.                                                                                                                                                 |
# | drive-wheels      | 4wd, fwd, rwd.                                                                                                                                                                                 |
# | engine-location   | front, rear.                                                                                                                                                                                   |
# | wheel-base        | continuous from 86.6 120.9.                                                                                                                                                                    |
# | length            | continuous from 141.1 to 208.1.                                                                                                                                                                |
# | width             | continuous from 60.3 to 72.3.                                                                                                                                                                  |
# | height            | continuous from 47.8 to 59.8.                                                                                                                                                                  |
# | curb-weight       | continuous from 1488 to 4066.                                                                                                                                                                  |
# | engine-type       | dohc, dohcv, l, ohc, ohcf, ohcv, rotor.                                                                                                                                                        |
# | num-of-cylinders  | eight, five, four, six, three, twelve, two.                                                                                                                                                    |
# | engine-size       | continuous from 61 to 326.                                                                                                                                                                     |
# | fuel-system       | 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.                                                                                                                                                  |
# | bore              | continuous from 2.54 to 3.94.                                                                                                                                                                  |
# | stroke            | continuous from 2.07 to 4.17.                                                                                                                                                                  |
# | compression-ratio | continuous from 7 to 23.                                                                                                                                                                       |
# | horsepower        | continuous from 48 to 288.                                                                                                                                                                     
# | peak-rpm          | continuous from 4150 to 6600.                                                                                                                                                                  |
# | city-mpg          | continuous from 13 to 49.                                                                                                                                                                      |
# | highway-mpg       | continuous from 16 to 54.                                                                                                                                                                      |
# | price             | continuous from 5118 to 45400.                                                                                                                                                                 |
# 
#       
# - `price` column is the one we want to predict.
# 
# ### Reading the Dataset

# In[36]:


import pandas as pd
import numpy as np

pd.options.display.max_columns = 99


# First of all we read our data set into a data frame, so that we can manipulate it easily.Working only on numeric columns those with continuous values.We can see from above mentioned attributes table which columns are continuous.

# In[37]:


cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)
cars.head()


# In[38]:


# Selecting only the columns with continuous values from - https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]
numeric_cars.head(5)


# ### Data Cleaning
# We can tell that the `normalized-losses` column contains missing values represented using "?". Let's replace these values and look for the presence of missing values in other numeric columns. Let's also rescale the values in the numeric columns so they all range from __0__ to __1__.

# In[39]:


numeric_cars = numeric_cars.replace('?', np.nan)
numeric_cars.head()


# Converting column to numeric 'float' type

# In[40]:


numeric_cars = numeric_cars.astype('float')
numeric_cars.isnull().sum()


# In[41]:


# Because `price` is the column we want to predict, let's remove any rows with missing `price` values.
numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars.isnull().sum()


# In[42]:


# Replacing missing values in other columns using column means.
numeric_cars = numeric_cars.fillna(numeric_cars.mean())


# In[43]:


# Confirming that there's no more missing values!
numeric_cars.isnull().sum()


# In[88]:


numeric_cars.shape


# ### Normalize
# Normalizing the numeric columns so all values range from 0 to 1. Except price column, as it is the target column.

# In[44]:


# Normalize all columnns to range from 0 to 1 except the target column.
price_col = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_col
numeric_cars.head()


# ### Univariate Model
# Starting with some univariate k-nearest neighbors models before moving to more complex models helps us structure your code workflow and understand the features better.

# In[45]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Creating a function "knn_train_test()" that encapsulates the training
# and simple validation process.This function has 3 parameters
def knn_train_test(train_col, target_col, df):
    knn = KNeighborsRegressor()
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # Fit a KNN model using default k value.
    knn.fit(train_df[[train_col]], train_df[target_col])
    
    # Make predictions using model.
    predicted_labels = knn.predict(test_df[[train_col]])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse
# empty list for RMSE values
rmse_results = {}
train_cols = numeric_cars.columns.drop('price')

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    rmse_results[col] = rmse_val

# Create a Series object from the dictionary so 
# we can easily view the results, sort, etc
rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()


# __Horsepower__ has the lowest rmse value and hence performed best using default k value.
# 
# Now testing the Univaritate model using the diffrent k_values = [1,3,5,7,9] values as a parameter. 

# In[46]:


def knn_train_test(train_col, target_col, df):
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [1,3,5,7,9]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[train_col]], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[[train_col]])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    k_rmse_results[col] = rmse_val

k_rmse_results


# Visualizing the results using a scatter plot or a line plot.

# In[52]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y, "", markersize=20)
    plt.xlabel('k value')
    plt.ylabel('RMSE')


# ### Multivariate Model
# Now we know that best K values is 5 . We write a function to accept multiple columns to train and test a multivariate k-nearest neighbors model using the default k value.
# 
# Steps:
# - Use the best 2 features(Columns) from the previous step.
# - Use the best 3 features(Columns) from the previous step.
# - Use the best 4 features(Columns) from the previous step.
# - Use the best 5 features(Columns) from the previous step.
# - Diplay all of the RMSE values

# In[48]:


# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
sorted_series_avg_rmse = series_avg_rmse.sort_values()
print(sorted_series_avg_rmse)

sorted_features = sorted_series_avg_rmse.index


# In[91]:


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [5]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

for nr_best_feats in range(2,7):
    k_rmse_results['{} best features'.format(nr_best_feats)] = knn_train_test(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

k_rmse_results


# ### Hyperparameter Tuning
# Now optimize the model that performed the best in the previous step. For the top 3 models in the last step, vary the hyperparameter value from 1 to 25 and then ploting the resulting RMSE values.

# In[50]:


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

for nr_best_feats in range(2,6):
    k_rmse_results['{} best features'.format(nr_best_feats)] = knn_train_test(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

k_rmse_results


# ### Ploting the resulting RMSE values

# In[51]:



for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')


# In[ ]:





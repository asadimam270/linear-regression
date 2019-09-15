#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  Linear Regression Project


# Stock market prediction
# 1.  Download HW3 data from the folder data in Documents on Blackboard, which lists daily signal and price values of S&P 500.
# 	
# 2.  The aim of this project is to test a data source (signal, second column in data.csv) which claims to be predictive of future returns of the SP500 index (spy_close_price, third column in data.csv).  We use SPY (SPDR S&P 500 ETF) as a proxy for the SP500 index. 
# 
# 3.  The signal and spy_close_price are both received at the same time at the end of the day on the date listed in column 1.  We do not know how the signal is generated or have a prior conviction about the forecast horizon over which the signal is supposed to be effective, nor its stationarity. 
# 
# 4.  The first step in this endeavor is data cleaning.  Assume all values in data.csv are potentially suspect, and please identify any errors in the data, flag them with a note, and suggest a corrected value or if advisable, you may choose to ignore them for purposes of your analysis.  Please explain what types of analysis you did to identify the errors, and provide any assumptions/intuition/formulas/scripts you may have used to help you find them.
# 
# 5.  Given the cleaned/censored version of the data you created in (4), please perform an analysis of the predictive power of signal with respect to spy_close_price.  This analysis must be based on linear regression. report the performace of linear regression model, scatter plots befor and after clearning the data. and explain if the signal can be a good predictive for the stock price. Also compute the correlation between the signal and price.
# 
# 
# Please document all experiment(s) you performed (including relevant code, package references, etc) and summarize your conclusions about the viability and shortcomings of this signal as a predictor of spy_close_price, including any materials you feel are appropriate to support your conclusions (eg, graphs, tables, etc).  Use this jupyter notebook file.
# 
# 
# 

# #  Importing the Data 

# In[4]:


# Import statements
import pandas as pd
import numpy as np

# Reading the file
filepath = '/Users/asadimam270/Downloads/data.csv'
data = pd.read_csv(filepath)
data.head()


#                         Separating independent and dependent variables

# In[5]:


# Dependent Variable
X = data["signal"].values

# Independent Variable 
Y = data["spy_close_price"].values


# # Visualizing the Uncleaned data

# In[25]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Scatterplot 
plt.scatter(X,Y)


# #                              Cleaning the Data for Analysis

#                                     Missing Value Test

# In[7]:


data.isnull().sum()


# ######                        Dealing With Outliers of the Dependent Variable

# In[8]:


# Visualizing outliers for Y
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# create box and whisker plot
plt.boxplot(Y)
# show line plot
plt.show()


# In[9]:


print("Replacing all elements of the dependent variable which are greater than 500 with mean")
Y[Y > 500] = Y.mean()
print(Y)


# In[10]:


# create box and whisker plot
plt.boxplot(Y)
# show line plot
plt.show()


# ######                        Dealing With Outliers of the Independent Variable

# In[24]:


# Visualizing outliers for X
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# create box and whisker plot
plt.boxplot(X)
# show line plot
plt.show()


# In[12]:


print("Replacing all elements of the independent variable which are greater than 100 with mean")
X[X > 100] = X.mean()
print(X)


# In[13]:


# create box and whisker plot
plt.boxplot(X)
# show line plot
plt.show()


# In[14]:


print("Now Replacing all elements of the independent variable which are less than 2 with mean")
X[X < 2] = X.mean()
print(X)


# In[15]:


# create box and whisker plot
plt.boxplot(X)
# show line plot
plt.show()


# # Performing Analysis on the Cleaned Data

# In[16]:


# Import Statements
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)


# In[18]:


# fit the regression model
regressor = LinearRegression()
regressor.fit(X_test.reshape(-1,1) ,Y_test)


# In[19]:


Y_predict = regressor.predict(X_test.reshape(-1,1))


# In[20]:


Y_predict


# ###### Calculating the mean square 

# In[21]:


# Calculating the mean square for Error 
mean_squared_error(Y_test,Y_predict)


# ###### Calculating the $R^2$ value

# In[22]:


r2_score(Y_test, Y_predict)  


# # Visualizing the model

# In[23]:


# Scatterplot 
plt.scatter(Y_test,Y_predict)


# The high $R^2$ value of 0.9670753789043635 and the small MSE value of 13.955863620130062 shows that this is a good model. This is also backed up the the visual result in scatterplot. 

# ### *Helpful Websites*

# <https://www.youtube.com/watch?v=NUXdtN1W1FE>

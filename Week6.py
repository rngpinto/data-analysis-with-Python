import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline


# Read the online file by the URL provides above, and assign it to variable "df"

df=pd.read_csv('kc_house_data_NaN.csv')


# show the first rows using dataframe.head() method

df.head(20)


#Exerc 1

df.dtypes




#Exerc 2

#Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop()
df.drop('id',axis = 1, inplace=True)
df.drop('Unnamed: 0',axis = 1, inplace=True)

#Describe (This method will provide various summary statistics, excluding NaN (Not a Number) values.)
#This shows the statistical summary of all numeric-typed (int, float) columns.

df.describe()

# describe all the columns in "df" 
df.describe(include = "all")


#select the columns of a dataframe by indicating the name of each column:
#e.g.:

df[['date']].describe()

# look at the info of "df"; permite identificar os missings; neste caso bedrooms e bathrooms
#waterfront and date are object types
#df.info()


df.describe()




#Module 2: Data Wrangling

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
df.info()


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())





#3 Exploratory Data Analysis

#Exerc3
#Use the method value_counts to count the number of houses with unique floor values, 
#use the method .to_frame() to convert it to a dataframe.


df[['floors']].value_counts()

df[['floors']].value_counts().to_frame()

#floors_counts.rename(columns={'floors':'value_counts'}, inplace=True)
#floors_counts


#Exer 4


#Import visualization packages "Matplotlib" and "Seaborn". Don't forget about "%matplotlib inline" to plot in a Jupyter notebook.

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 


df[['waterfront']].describe()

sns.boxplot(x="waterfront", y="price", data=df)



#Exer 5

sns.regplot(x="sqft_above", y="price", data=df)

df[["sqft_above", "price"]].corr()



#Module 4: Model Development


#Exer 6

X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)



#We can output a prediction:
Yhat=lm.predict(X)

Yhat[0:5]


#What is the value of the intercept (a)?
lm.intercept_

#What is the value of the slope (b)?
lm.coef_


#Exer 7

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] 


Z=df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]     

lm.fit(Z, df['price'])

print('The R-square is: ', lm.score(Z, df['price']))



#Exerc 8
#We create the pipeline by creating a list of tuples 
#including the name of the model or estimator and its corresponding constructor.

from sklearn.preprocessing import PolynomialFeatures
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


#We input the list as an argument to the pipeline constructor:

pipe=Pipeline(Input)
pipe


#We will also fit the object using the features in the list features, and calculate the R^2. 

pipe.fit(df[features],df['price'])
pipe.score(df[features],df['price'])



#Exerc 9: 


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


#Create and fit a Ridge regression object using the training data, 
#set the regularization parameter to 0.1, and calculate the R^2 using the test data.

from sklearn.linear_model import Ridge

#Let's create a Ridge regression object, setting the regularization parameter (alpha) to 0.1

RigeModel=Ridge(alpha=0.1)

#Like regular regression, you can fit the model using the method fit.

RigeModel.fit(x_train, y_train)

#calculate the R^2 using the test data

RigeModel.score(x_test, y_test)



from sklearn.preprocessing import PolynomialFeatures


pr = PolynomialFeatures(degree=2)

#Performing a 2nd order polynomial transformation:

x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)


#Creating a Ridge regression
RigeModel=Ridge(alpha=0.1)

#Fitting the regression
RigeModel.fit(x_train_pr, y_train)

#R^2 calculation:
RigeModel.score(x_test_pr, y_test)


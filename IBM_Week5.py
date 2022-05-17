import pandas as pd
import numpy as np


df = pd.read_csv('IBM_DA/auto_week4.csv')

df.head()

df.drop('Unnamed: 0', axis = 1, inplace=True)
 
df.head()



#First, let's only use numeric data:

df=df._get_numeric_data()
df.head()


#Libraries for plotting:

from ipywidgets import interact, interactive, fixed, interact_manual



#Functions for Plotting:

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()



def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()




#Part 1: Training and Testing

#An important step in testing your model is to split your data into training and testing data. 
#We will place the target data price in a separate dataframe y_data:

y_data = df['price']

#Drop price data in dataframe x_data:

x_data=df.drop('price',axis=1)



#Now, we randomly split our data into training and testing data using the function train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])



#The test_size parameter sets the proportion of data that is split into the testing set. 
#In the above, the testing set is 10% of the total dataset.


#Let's import LinearRegression from the module linear_model.

from sklearn.linear_model import LinearRegression



#We create a Linear Regression object:
lre=LinearRegression()


#We fit the model using the feature "horsepower":
lre.fit(x_train[['horsepower']], y_train)


#Let's calculate the R^2 on the test data:
lre.score(x_test[['horsepower']], y_test)


#We can see the R^2 is much smaller using the test data compared to the training data.
lre.score(x_train[['horsepower']], y_train)



#Sometimes you do not have sufficient testing data; as a result, you may want to perform cross-validation. 
#Let's go over several methods that you can use for cross-validation.


#Cross-Validation Score
#Let's import model_selection from the module cross_val_score.

from sklearn.model_selection import cross_val_score


#We input the object, the feature ("horsepower"), and the target data (y_data). 
#The parameter 'cv' determines the number of folds. In this case, it is 4.

Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

#The default scoring is R^2. Each element in the array has the average R^2 value for the fold:

Rcross


#We can calculate the average and standard deviation of our estimate:

print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())



#We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.

-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')



#You can also use the function 'cross_val_predict' to predict the output. 
#The function splits up the data into the specified number of folds, with one fold for testing 
#and the other folds are used for training. First, import the function:

from sklearn.model_selection import cross_val_predict



#We input the object, the feature "horsepower", and the target data y_data. 
#The parameter 'cv' determines the number of folds. In this case, it is 4. We can produce an output:


yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]


#Part 2: Overfitting, Underfitting and Model Selection



#Let's create Multiple Linear Regression objects and train the model using 
#'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features.

lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)


#Prediction using training data:

yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]


#Prediction using test data:

yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]


#Let's perform some model evaluation using our training and testing data separately. 
#First, we import the seaborn and matplotlib library for plotting.


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


#Let's examine the distribution of the predicted values of the training data.

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)


#So far, the model seems to be doing well in learning from the training dataset. 
#But what happens when the model encounters new data from the testing dataset? 
#When the model generates new values from the test data, we see the distribution of the predicted values is much different 
#from the actual target values.


Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)


#Comparing Figure 1 and Figure 2, 
#it is evident that the distribution of the test data in Figure 1 is much better at fitting the data.

# Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset.

from sklearn.preprocessing import PolynomialFeatures


#Overfitting

#Overfitting occurs when the model fits the noise, but not the underlying process. 
#Therefore, when testing your model using the test set, your model does not perform as well since it is modelling noise, 
#not the underlying process that generated the relationship. Let's create a degree 5 polynomial model.

#Let's use 55 percent of the data for training and the rest for testing:

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)


#We will perform a degree 5 polynomial transformation on the feature 'horsepower'.

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr


#Now, let's create a Linear Regression model "poly" and train it.

poly = LinearRegression()
poly.fit(x_train_pr, y_train)



#We can see the output of our model using the method "predict." We assign the values to "yhat".

yhat = poly.predict(x_test_pr)
yhat[0:5]



#Let's take the first five predicted values and compare it to the actual targets.

print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)

#We will use the function "PollyPlot" 
#that we defined at the beginning of the lab to display the training data, testing data, and the predicted function.




#Figure 3: A polynomial regression model where red dots represent training data,
#green dots represent test data, and the blue line represents the model prediction.



#We see that the estimated function appears to track the data but around 200 horsepower, the function begins to diverge from the data points.

#R^2 of the training data:

poly.score(x_train_pr, y_train)


#R^2 of the test data:

poly.score(x_test_pr, y_test)


#We see the R^2 for the training data is 0.5567 while the R^2 on the test data was -29.87. 
#The lower the R^2, the worse the model. A negative R^2 is a sign of overfitting.

PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)

#Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')



#We see the R^2 gradually increases until an order three polynomial is used. 
#Then, the R^2 dramatically decreases at an order four polynomial.
        

#The following function will be used in the next section. Please run the cell below.
        
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)



#The following interface allows you to experiment with different polynomial orders and different amounts of data.

interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))


#Exerc: Create a "PolynomialFeatures" object "pr1" of degree two.
pr1=PolynomialFeatures(degree=2)


#Exerc: Transform the training and testing samples for the features 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'. 
#Hint: use the method "fit_transform".

x_train_pr1=pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1=pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])


#Exerc: How many dimensions does the new feature have? Hint: use the attribute "shape".
x_train_pr1.shape #there are now 15 features


#Exerc: Create a linear regression model "poly1". Train the object using the method "fit" using the polynomial features.
poly1=LinearRegression().fit(x_train_pr1,y_train)


#Exerc: Use the method "predict" to predict an output on the polynomial features, 
#then use the function "DistributionPlot" to display the distribution of the predicted test output vs. the actual test data.

yhat_test1=poly1.predict(x_test_pr1)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)

#The predicted value is higher than actual value for cars where the price $10,000 range, conversely the predicted price is lower than the price cost in the $30,000 to $40,000 range. As such the model is not as accurate in these ranges.




#Part 3: Ridge Regression

#In this section, we will review Ridge Regression and see how the parameter alpha changes the model. 
#Just a note, here our test data will be used as validation data.

#Let's perform a degree two polynomial transformation on our data.

pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])



#Let's import Ridge from the module linear models.

from sklearn.linear_model import Ridge

#Let's create a Ridge regression object, setting the regularization parameter (alpha) to 0.1

RigeModel=Ridge(alpha=1)

#Like regular regression, you can fit the model using the method fit.

RigeModel.fit(x_train_pr, y_train)


#Similarly, you can obtain a prediction:

yhat = RigeModel.predict(x_test_pr)


#Let's compare the first five predicted samples to our test set:

print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)



#We select the value of alpha that minimizes the test error. To do so, we can use a for loop. 
#We have also created a progress bar to see how many iterations we have completed so far.


from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


#We can plot out the value of R^2 for different alphas:

width = 12
height = 10
plt.figure(figsize=(width, height))
plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()


#Exerc: Calculate the R^2 using the polynomial features, use the training data to train the model and use the test data to test the model. 
#The parameter alpha should be set to 10.


RigeModel = Ridge(alpha=10) 
RigeModel.fit(x_train_pr, y_train)
RigeModel.score(x_test_pr, y_test)




#Part 4: Grid Search
#The term alpha is a hyperparameter. Sklearn has the class GridSearchCV to make the process of finding the best hyperparameter simpler.

#Let's import GridSearchCV from the module model_selection.

from sklearn.model_selection import GridSearchCV


#We create a dictionary of parameter values:

parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
parameters1


#Create a Ridge regression object:

RR=Ridge()
RR


#Create a ridge grid search object:

Grid1 = GridSearchCV(RR, parameters1,cv=4)


#Fit the model:
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)


#The object finds the best parameter values on the validation data. 
BestRR=Grid1.best_estimator_
BestRR


#We now test our model on the test data:

BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)



#Save the new csv:
 
df.to_csv('IBM_DA/auto_week5.csv')


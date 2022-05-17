
#   WEEK 3

#Analyzing Individual Feature Patterns Using Visualization


import pandas as pd
import numpy as np

df = pd.read_csv('auto_week2.csv')

df.head()

df.drop('Unnamed: 0', axis = 1, inplace=True)

df.head()



#2. Analyzing Individual Feature Patterns Using Visualization


#Import visualization packages "Matplotlib" and "Seaborn". Don't forget about "%matplotlib inline" to plot in a Jupyter notebook.

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 


#How to choose the right visualization method?


# list the data types for each column
print(df.dtypes)

df.drop('highway-L/100km', axis = 1, inplace=True)

#Convert data types to proper format:
df[["horsepower"]] = df[["horsepower"]].astype("float")
print(df.dtypes)

df.head()


#For example, we can calculate the correlation between variables of type "int64" or "float64" using the method "corr":

df.corr()



#Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower.

df[['bore','stroke','compression-ratio','horsepower']].corr()


#Positive Linear Relationship 
#Let's find the scatterplot of "engine-size" and "price".

# Engine size as potential predictor variable of price
#sns.regplot(x="engine-size", y="price", data=df)
#plt.ylim(0,)


#We can examine the correlation between 'engine-size' and 'price' and see that it's approximately 0.87.

df[["engine-size", "price"]].corr()


#Highway mpg is a potential predictor variable of price. Let's find the scatterplot of "highway-mpg" and "price".

#sns.regplot(x="highway-mpg", y="price", data=df)
#plt.ylim(0,)


#We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704.
df[["highway-mpg", "price"]].corr()



#Weak Linear Relationship

#Let's see if "peak-rpm" is a predictor variable of "price".

#sns.regplot(x="peak-rpm", y="price", data=df)
#plt.ylim(0,)

df[['peak-rpm','price']].corr()



#Find the correlation between x="stroke" and y="price".
df[['stroke','price']].corr()


#sns.regplot(x="stroke", y="price", data=df)
#plt.ylim(0,)



#Categorical Variables


#A good way to visualize categorical variables is by using boxplots.

#Let's look at the relationship between "body-style" and "price".

#sns.boxplot(x="body-style", y="price", data=df)
#sns.boxplot(x="engine-location", y="price", data=df)


# drive-wheels
#sns.boxplot(x="drive-wheels", y="price", data=df)


#Here we see that the distribution of price between the different drive-wheels categories differs. As such, drive-wheels could potentially be a predictor of price.


#3. Descriptive Statistical Analysis
#The describe function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.

df.describe()

#The default setting of "describe" skips variables of type object. 
#We can apply the method "describe" on the variables of type 'object' as follows:


df.describe(include=['object'])


#Value Counts

#Value counts is a good way of understanding how many units of each characteristic/variable we have. We can apply the "value_counts" method on the column "drive-wheels". Don’t forget the method "value_counts" only works on pandas series, not pandas dataframes. As a result, we only include one bracket df['drive-wheels'], not two brackets df[['drive-wheels']].

df['drive-wheels'].value_counts()


#We can convert the series to a dataframe as follows:

df['drive-wheels'].value_counts().to_frame()

#Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts


#Now let's rename the index to 'drive-wheels':

drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts



#4. Basics of Grouping


#For example, let's group by the variable "drive-wheels". We see that there are 3 different categories of drive wheels.
df['drive-wheels'].unique()


#We can select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".

df_group_one = df[['drive-wheels','body-style','price']]


#We can then calculate the average price for each of the different categories of data.


df.round(decimals=6)

# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one



#You can also group by multiple variables. For example, let's group by both 'drive-wheels' and 'body-style'. This groups the dataframe by the unique combination of 'drive-wheels' and 'body-style'. We can store the results in the variable 'grouped_test1'.

# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1



#This grouped data is much easier to visualize when it is made into a pivot table. A pivot table is like an Excel spreadsheet, with one variable along the column and another along the row. We can convert the dataframe to a pivot table using the method "pivot" to create a pivot table from the groups.
#In this case, we will leave the drive-wheels variable as the rows of the table, and pivot body-style to become the columns of the table:

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


#Often, we won't have data for some of the pivot cells. We can fill these missing cells with the value 0, but any other value could potentially be used as well. It should be mentioned that missing data is quite a complex subject and is an entire course on its own.

grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot



#EXERC
#Use the "groupby" function to find the average "price" of each car based on "body-style".


# grouping results
#df_gptest = df[['body-style','price']]
#grouped_test = df_gptest.groupby(['body-style'],as_index=False).mean()
#grouped_test


#Variables: Drive Wheels and Body Style vs. Price

#Let's use a heat map to visualize the relationship between Body Style vs Price.

#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()



#The default labels convey no useful information to us. Let's change that:

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


from scipy import stats

#Wheel-Base vs. Price
#Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


#Conclusion:
#Since the p-value is < 0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585).


#ANOVA: Analysis of Variance

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)

#We can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.
# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)


#Save the new csv:

df.to_csv('auto_week3.csv')


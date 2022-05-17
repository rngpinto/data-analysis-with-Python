

# import pandas library
import pandas as pd
import numpy as np


# Read the online file by the URL provides above, and assign it to variable "df"
df = pd.read_csv('auto.csv', header=None)


# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
df.head(5)

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]



df = pd.read_csv('auto.csv', names = headers)


# To see what the data set looks like, we'll use the head() method.
df.head()


#Identify and handle missing values


#Convert "?" to NaN
import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)



#We use the following functions to identify these missing values.

missing_data = df.isnull()
missing_data.head(5)



#Count missing values in each column

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


#Based on the summary above, each column has 205 rows of data and seven of the columns containing missing data:

#"normalized-losses": 41 missing data
#"num-of-doors": 2 missing data
#"bore": 4 missing data
#"stroke" : 4 missing data
#"horsepower": 2 missing data
#"peak-rpm": 2 missing data
#"price": 4 missing data




#Deal with missing data

#Calculate the mean value for the "normalized-losses" column 
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


#Replace "NaN" with mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)



#Calculate the mean value for the "bore" column
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

#Replace "NaN" with the mean value in the "bore" column
df["bore"].replace(np.nan, avg_bore, inplace=True)

df.head(5)


#Calculate the mean value for the "stroke" column 
avg_strk = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_strk)


#Replace "NaN" with mean value in "normalized-losses" column
df["stroke"].replace(np.nan, avg_strk, inplace=True)



avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)



#To see which values are present in a particular column, we can use the ".value_counts()" method:
df['num-of-doors'].value_counts()

#or:

df['num-of-doors'].value_counts().idxmax()


#The replacement procedure is very similar to what we have seen previously:
df["num-of-doors"].replace(np.nan, "four", inplace=True)


# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


df.head()





#Correct data format¶

#The last step in data cleaning is checking and making sure that all data is in the correct format (int, float, text or other).

#In Pandas, we use:
#.dtype() to check the data type
#.astype() to change the data type


#Let's list the data types for each column
df.dtypes



df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")



#Let us list the columns after the conversion
df.dtypes





#Data Standardization


df.head()

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['highway-L/100km'] = 235/df["highway-mpg"]

# check your transformed data 
df.head()


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
df.head()




#Data Normalization:


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

df.head()






#Example of Binning Data In Pandas


#Convert data to correct format:

df["horsepower"]=df["horsepower"].astype(int, copy=True)


#Let's plot the histogram of horsepower to see what the distribution of horsepower looks like.

%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


#We would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated function.
#Since we want to include the minimum value of horsepower, we want to set start_value = min(df["horsepower"]).
#Since we want to include the maximum value of horsepower, we want to set end_value = max(df["horsepower"]).
#Since we are building 3 bins of equal length, there should be 4 dividers, so numbers_generated = 4.

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


#We set group names:
group_names = ['Low', 'Medium', 'High']


#We apply the function "cut" to determine what each value of df['horsepower'] belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


#Let's see the number of vehicles in each bin:
df["horsepower-binned"].value_counts()


#Bins Visualization¶

#Let's plot the distribution of each bin:

%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())


# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


#Indicator Variable (or Dummy Variable)

#We see the column "fuel-type" has two unique values: "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, we convert "fuel-type" to indicator variables.

#We will use pandas' method 'get_dummies' to assign numerical values to different categories of fuel type.


df.columns


#Get the indicator variables and assign it to data frame "dummy_variable_1":
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()

#Change the column names for clarity:

dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()


#In the dataframe, column 'fuel-type' has values for 'gas' and 'diesel' as 0s and 1s now.

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)


# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
df.head()


#Save the new csv:

df.to_csv('auto_week2.csv')

df.head()




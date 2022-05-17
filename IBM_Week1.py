import download

# import pandas library
import pandas as pd
import numpy as np

path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"


# Import pandas library
import pandas as pd


# Read the online file by the URL provides above, and assign it to variable "df"
df = pd.read_csv(path, header=None)

# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
df.head(5)


# Write your code below and press Shift+Enter to execute 
print("The bottom 10 rows of the dataframe") 
df.tail(10)



# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)


#replace headers and recheck our dataframe:

df.columns = headers
df.head(10)


#We need to replace the "?" symbol with NaN so the dropna() can remove the missing values:

df1=df.replace('?',np.NaN)

df.head(10)

#We can drop missing values along the column "price" as follows:
df=df1.dropna(subset=["price"], axis=0)
df.head(20)

# Find the name of the columns of the dataframe.
print(df.columns)


df.to_csv("automobile.csv", index=False)



#Basic Insight of Dataset

#Data types:

# check the data type of data frame "df" by .dtypes
print(df.dtypes)


#Describe (This method will provide various summary statistics, excluding NaN (Not a Number) values.)
#This shows the statistical summary of all numeric-typed (int, float) columns.

df.describe()


# describe all the columns in "df" 
df.describe(include = "all")



#select the columns of a dataframe by indicating the name of each column:
#e.g.:

df[['length','compression-ratio']].describe()


#Info:
#This method prints information about a DataFrame, 
#including the index dtype and columns, non-null values and memory usage.

# look at the info of "df"
df.info()
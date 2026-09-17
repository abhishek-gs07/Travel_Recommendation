import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import flask

data=pd.read_csv("Worldwide Travel Cities Dataset (Ratings and Climate).csv")

#Checked all null values, missing values and duplicated values
"""print(data.head(),"\n")
print(data.tail(),"\n")
print(data.info(),"\n")
print(data.isnull().sum(),"\n")
print(data.duplicated().sum(),"\n")
print(data.describe(include=object)) #include object forces panda to include text data
"""

#level all the numeric values to 0-1 scaling level
budget_map={"Budget":1,"Mid-range":3,"Luxury":5}
data["budget_level"]=data["budget_level"].map(budget_map)
#print(data["budget_level"])
duration_map={"Day trip":1,"Short trip":2,"Weekend":3,"One week":4,"Long trip":5}
#print(data["ideal_durations"])

scaler=MinMaxScaler()
numeric_columns=data.select_dtypes(include=["int64","float64"]).columns
data[numeric_columns]=scaler.fit_transform(data[numeric_columns])

print(data["region"].unique())

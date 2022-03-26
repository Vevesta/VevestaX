#!/usr/bin/env python
# coding: utf-8

#import the vevesta Library
from vevestaX import vevesta as v

#create a vevestaX object
V=v.Experiment()

#read the dataset
import pandas as pd
df=pd.read_csv("data.csv")

print(df.head(2))

#Extract the columns names for features
V.ds=df
#you can also use:
#V.dataSourcing = df

#Print the feature being used
print(V.ds)

# Do some feature engineering
df["salary_feature"]= df["Salary"] * 100/ df["House_Price"]
df['salary_ratio1']=df["Salary"] * 100 / df["Months_Count"] * 100

#Extract features engineered
V.fe=df

#you can also use:
#V.featureEngineering = df


#Print the features engineered
print(V.fe)

#Track variables which have been used for modelling
V.start()

#you can also use:
#V.startModelling()

#All the varibales mentioned here will be tracked
epochs=1500
seed=2000
loss='rmse'
accuracy= 91.2

#end tracking of variables
V.end()
#you can also use V.endModelling()


V.start()
recall = 95
precision = 87
V.end()

# Dump the datasourcing, features engineered and the variables tracked in a xlsx file
V.dump(techniqueUsed='XGBoost',filename="vevestaDump.xlsx",message="precision is tracked",version=1)

#if filename is not mentioned, then by default the data will be dumped to vevesta.xlsx file
#V.dump(techniqueUsed='XGBoost')


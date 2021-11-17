#import the vevesta Library
from vevestaX import vevesta as v
V=v.Experiment()


#read the dataset
import pandas as pd
df=pd.read_csv("salaries.csv")
df.head(2)

#Extract the columns names for features
V.ds=df
# you can also use:
#   V.datasourcing = df



#Print the feature being used
V.ds


# Do some feature engineering
df["age"]=50
df['gender']='F'


#Extract features engineered
V.fe=df  
# you can also use:
#  V.featureEngineering = df


#Print the features engineered
V.fe


#Track variables which have been used for modelling
V.start()
# you can also use:
#  V.startModelling()


# All the varibales mentioned here will be tracked
epochs=100
seed=3
loss='rmse'


#end tracking of variables
V.end()
# you can also use V.endModelling()


# Dump the datasourcing, features engineered and the variables tracked in a xlsx file
V.dump(techniqueUsed='XGBoost',filename="vevestaDump1.xlsx",message="no values",version=1)

#if filename is not mentioned, then by default the data will be dumped to vevesta.xlsx file


'''
After using V.dump four sheets will be created in the mentioned xlsx file.
 The four sheets will contain following details:
    

Sheet1: Data Sourcing 
Sheet2: Features Engineered 
Sheet3: Data Modelling 
Sheet4: Message

'''


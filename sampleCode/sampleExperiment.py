#import the vevesta Library
from vevestaX import vevesta as v

#create a vevestaX object
V=v.Experiment()


#read the dataset
import pandas as pd
df=pd.read_csv("salaries.csv")
print(df.head(2))

#Extract the columns names for features
V.ds=df
#you can also use:
#V.datasourcing = df



#Print the feature being used
print(V.ds)


# Do some feature engineering
df["age"]=50
df['gender']='F'


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


# All the varibales mentioned here will be tracked
epochs=150
seed=3
loss='rmse'
accuracy=98.7


#end tracking of variables
V.end()
# you can also use V.endModelling()


# Dump the datasourcing, features engineered and the variables tracked in a xlsx file
V.dump(techniqueUsed='XGBoost',filename="vevestaDump1.xlsx",message="accuracy increased",version=1)

#if filename is not mentioned, then by default the data will be dumped to vevesta.xlsx file
#V.dump(techniqueUsed='XGBoost')


'''
After using V.dump four sheets will be created in the mentioned xlsx file.
 The four sheets will contain following details:
    

Sheet1: Data Sourcing 
Sheet2: Features Engineered 
Sheet3: Data Modelling 
Sheet4: Message

'''

#Sheet 1 for datasourcing
data_ds=pd.read_excel("vevestaDump1.xlsx",'dataSourcing')
print(data_ds)

#Sheet 2 for featuresEngineered
data_fe=pd.read_excel("vevestaDump1.xlsx",'featureEngineering')
print(data_fe)

#Sheet 3 for dataModelling
data_mod=pd.read_excel("vevestaDump1.xlsx",'modelling')
print(data_mod)

#Sheet 4 for message
data_msg=pd.read_excel("vevestaDump1.xlsx",'messages')
print(data_msg)
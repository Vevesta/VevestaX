# VevestaX

![image](https://user-images.githubusercontent.com/81908188/142753559-9f94639b-324b-4734-a183-cd7d2c97a3fc.png)

[![Downloads](https://static.pepy.tech/personalized-badge/vevestax?period=total&units=international_system&left_color=blue&right_color=grey&left_text=Downloads)](https://pepy.tech/project/vevestax) [![Downloads](https://static.pepy.tech/personalized-badge/vevestax?period=total&units=international_system&left_color=orange&right_color=grey&left_text=Retweet)](https://twitter.com/vevesta1/status/1503747980188594178?s=20&t=3zXxSDS8WCddWcQHDxUrtg) 


# Track failed and successful Machine Learning experiments as well as features.

VevestaX is an open source Python package for ML Engineers and Data Scientists.  It includes modules for tracking features sourced from data, feature engineering and variables. The output is an excel file which has tabs namely, data sourcing, feature engineering and modelling. The library can be used with Jupyter notebook, IDEs like spyder or while running the python script through command line. VevestaX is framework agnostic. You can use it with any machine learning or deep learning framework.



## How to install the library:
```
pip install vevestaX
```
## How to import a library and create the object
```
#import the vevesta Library
from vevestaX import vevesta as v
V=v.Experiment()
```


## How to extract features present in input data.
![image](https://user-images.githubusercontent.com/81908188/141691820-511ebba1-bc5a-4ce4-acd0-cd23ae3cd782.png)
Code snippet:
```
#read the dataset
import pandas as pd
df=pd.read_csv("salaries.csv")
df.head(2)

#Extract the columns names for features
V.ds=df
# you can also use:
#   V.dataSourcing = df

#Print the feature being used
V.ds
```

## How to extract engineered features
![image](https://user-images.githubusercontent.com/81908188/140041279-7ecd6444-a9ba-4e87-a0e5-46435c759d18.png)

Code snippet
```
#Extract features engineered
V.fe=df  
# you can also use:
V.featureEngineering = df

#Print the features engineered
V.fe
```
## How to track variables used in the code.
V.start() and V.end() form a code block and can be called multiple times in the code to track variables used within the code block. Any technique such as XGBoost, decision tree, etc can be used within this code block.
![image](https://user-images.githubusercontent.com/81908188/140041422-97be7287-111d-40c3-bc8f-d921db90acf8.png)
Code snippet:
```
#Track variables which have been used for modelling
V.start()
# you can also use: V.startModelling()


# All the variables mentioned here will be tracked
epochs=100
seed=3
loss='rmse'


#end tracking of variables
V.end()
# or, you can also use : V.endModelling()
```
## How to dump the features and modelling variables in an given excel file
![image](https://user-images.githubusercontent.com/81908188/140653881-1698d7ba-1c0f-4879-8a96-a90123108165.png)
Code snippet:
```
# Dump the datasourcing, features engineered and the variables tracked in a xlsx file
V.dump(techniqueUsed='XGBoost',filename="vevestaDump1.xlsx",message="XGboost with data augmentation was used",version=1)
```

Alternatively, write the experiment into the default file, vevesta.xlsx
![image](https://user-images.githubusercontent.com/81908188/140653897-6654e94b-a332-49a2-a7b7-416cb5bded5c.png)
Code snippet:
```
V.dump(techniqueUsed='XGBoost')
```
A sample output excel file has been uploaded on google sheets. Its url is [here](https://docs.google.com/spreadsheets/d/1iOL3jiiQ834_vep5E4fPpxj7QDGVxOBJ/edit?usp=sharing&ouid=103382336064969333270&rtpof=true&sd=true)

If you liked the library, please give us a github star and [retweet](https://twitter.com/vevesta1/status/1503747980188594178?s=20&t=3zXxSDS8WCddWcQHDxUrtg) .

For additional features, explore our tool at [Vevesta](https://www.vevesta.com) . For comments, suggestions and early access to the tool, reach out at vevestax@vevesta.com

We at vevesta Labs are maintaining this library and we welcome feature requests. Find detailed blog on the vevestaX on [Medium](https://medium.com/@priyanka_60446/vevestax-open-source-library-to-track-failed-and-successful-machine-learning-experiments-and-data-8deb76254b9c)

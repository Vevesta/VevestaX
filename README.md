# VevestaX

![image](https://user-images.githubusercontent.com/81908188/142753559-9f94639b-324b-4734-a183-cd7d2c97a3fc.png)

[![Downloads](https://static.pepy.tech/personalized-badge/vevestax?period=total&units=international_system&left_color=blue&right_color=grey&left_text=Downloads)](https://pepy.tech/project/vevestax) [![Downloads](https://pepy.tech/badge/vevestax/month)](https://pepy.tech/project/vevestax) [![Downloads](https://pepy.tech/badge/vevestax/week)](https://pepy.tech/project/vevestax) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fvevesta1&label=Retweet)](https://twitter.com/vevesta1/status/1503747980188594178?s=20&t=3zXxSDS8WCddWcQHDxUrtg)



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
A sample output excel file has been uploaded on google sheets. Its url is [here](https://docs.google.com/spreadsheets/d/1hs08dZfZ8bSfU1Aem_sX2aaoNYlslRBU/edit?usp=sharing&ouid=103382336064969333270&rtpof=true&sd=true)

## Output snapshots

### Sourced Data tab
![image](https://user-images.githubusercontent.com/81908188/160064875-7642a251-0c10-4acf-97b1-a9bde1fb0b0f.png)

### Feature Engineering tab
![image](https://user-images.githubusercontent.com/81908188/160064990-570e3a10-e13f-45e6-b089-7e1aa99e24b9.png)

### Modelling tab
![image](https://user-images.githubusercontent.com/81908188/160065199-7030314b-fcaf-4a53-91d1-f47b6725b4c7.png)

### Messages tab
![image](https://user-images.githubusercontent.com/81908188/160065290-a04d8f96-496a-4567-8c4b-f9a4514c2ba3.png)

### EDA-correlation tab
![image](https://user-images.githubusercontent.com/81908188/160065377-6a3b5695-d497-4fc3-921d-2e9b8f28dced.png)

### Experiments performance plots
![image](https://user-images.githubusercontent.com/81908188/160065501-c3b8a1c3-e75b-41fa-abea-cad3bb4b5add.png)
![image](https://user-images.githubusercontent.com/81908188/160065687-17821d15-2b12-4fc2-9978-f55d26c37ed0.png)



If you liked the library, please give us a github star and [retweet](https://twitter.com/vevesta1/status/1503747980188594178?s=20&t=3zXxSDS8WCddWcQHDxUrtg) .

For additional features, explore our tool at [Vevesta](https://www.vevesta.com) . For comments, suggestions and early access to the tool, reach out at vevestax@vevesta.com

Looking for beta users for the library. Register [here](https://forms.gle/wM1GKyYS7fDTxmS56)

We at vevesta Labs are maintaining this library and we welcome feature requests. Find detailed blog on the vevestaX on [Medium](https://medium.com/@priyanka_60446/vevestax-open-source-library-to-track-failed-and-successful-machine-learning-experiments-and-data-8deb76254b9c)

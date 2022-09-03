# VevestaX

![image](https://user-images.githubusercontent.com/81908188/142753559-9f94639b-324b-4734-a183-cd7d2c97a3fc.png)

[![Downloads](https://static.pepy.tech/personalized-badge/vevestax?period=total&units=international_system&left_color=blue&right_color=grey&left_text=Downloads)](https://pepy.tech/project/vevestax) [![Downloads](https://pepy.tech/badge/vevestax/month)](https://pepy.tech/project/vevestax) [![Downloads](https://pepy.tech/badge/vevestax/week)](https://pepy.tech/project/vevestax) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fvevesta1&label=Retweet)](https://twitter.com/vevesta1/status/1503747980188594178?s=20&t=3zXxSDS8WCddWcQHDxUrtg)



# Library to track ML experiments, do extensive EDA as well as GitHub checkins all in 2 lines of code
VevestaX is an open source Python package for ML Engineers and Data Scientists.  It does the following:

* Automatic EDA on the data
* ML Experiment tracking: Tracking features sourced from data, feature engineered and variables used over multiple experiments
* Checking the code into Github after every experiment runs. 

The output is an excel file. The library can be used with Jupyter notebook, IDEs like spyder, Colab, Kaggle notebook or while running the python script through command line. VevestaX is framework agnostic. You can use it with any machine learning or deep learning framework.

## Table of Contents
1. [How to Install VevestaX](https://github.com/Vevesta/VevestaX/blob/main/README.md#how-to-install-VevestaX)
2. [How to import VevestaX and create the experiment object](https://github.com/Vevesta/VevestaX/blob/main/README.md#How-to-import-VevestaX-and-create-the-experiment-object)
3. [How to extract features present in input pandas/pyspark dataframe](https://github.com/Vevesta/VevestaX/blob/main/README.md#How-to-extract-features-present-in-input-pandas-or-pyspark-dataframe)
4. [How to extract engineered features](https://github.com/Vevesta/VevestaX/blob/main/README.md#How-to-extract-engineered-features)
5. [How to track variables used](https://github.com/Vevesta/VevestaX/blob/main/README.md#How-to-track-variables-used)
6. [How to track all variables in the code while writing less code](https://github.com/Vevesta/VevestaX/blob/main/README.md#how-to-track-all-variables-in-the-code-while-writing-less-code)
7. [How to write the features and modelling variables in an given excel file](https://github.com/Vevesta/VevestaX/blob/main/README.md#How-to-write-the-features-and-modelling-variables-in-an-given-excel-file)
8. [How to commit file, features and parameters to Vevesta](https://github.com/Vevesta/VevestaX/blob/main/README.md#how-to-commit-file-features-and-parameters-to-vevesta)
9. [How to speed up the code i.e Skip EDA on every run](https://github.com/Vevesta/VevestaX/blob/main/README.md#how-to-speed-up-the-code)
10. [How to check in code into GitHub](https://github.com/Vevesta/VevestaX/blob/main/README.md#how-to-check-code-into-github)
11. [Snapshots of output excel file](https://github.com/Vevesta/VevestaX/blob/main/README.md#Snapshots-of-output-excel-file)

## How to install VevestaX
```
pip install vevestaX
```
## How to import VevestaX and create the experiment object
```
#import the vevesta Library
from vevestaX import vevesta as v
V=v.Experiment()
```


## How to extract features present in input pandas or pyspark dataframe
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
```

## How to extract engineered features
![image](https://user-images.githubusercontent.com/81908188/140041279-7ecd6444-a9ba-4e87-a0e5-46435c759d18.png)

Code snippet
```
#Extract features engineered
V.fe=df  
# you can also use:
V.featureEngineering = df
```
## How to track variables used
V.start() and V.end() form a code block and can be called multiple times in the code to track variables used within the code block. Any technique such as XGBoost, decision tree, etc can be used within this code block. All computed variables will be tracked between V.start() and V.end(). If V.start() and V.end() is not used, all the variables used in the code will be tracked.

Code snippet:
```
#Track variables which have been used for modelling
V.start()
# you can also use: V.startModelling()


# All the variables mentioned here will be tracked
epochs=100
seed=3
accuracy = computeAccuracy() #this will be computed variable
recall = computeRecall() #This will be computed variable
loss='rmse'


#end tracking of variables
V.end()
# or, you can also use : V.endModelling()
```

## How to track all variables in the code while writing less code
You can absolutely eliminate using V.start() and V.end() function calls. All the primitive data type variables used in the code are tracked and written to the excel file by default. Note: while on colab or kaggle, V.start() and V.end() feature hasn't been rolled out. Instead all the variables used in the code are tracked by default.

## How to write the features and modelling variables in an given excel file
![image](https://user-images.githubusercontent.com/81908188/140653881-1698d7ba-1c0f-4879-8a96-a90123108165.png)
Code snippet:
```
# Dump the datasourcing, features engineered and the variables tracked in a xlsx file
V.dump(techniqueUsed='XGBoost',filename="vevestaDump1.xlsx",message="XGboost with data augmentation was used",version=1, repoName='My_Project')
```

Alternatively, write the experiment into the default file, vevesta.xlsx
![image](https://user-images.githubusercontent.com/81908188/140653897-6654e94b-a332-49a2-a7b7-416cb5bded5c.png)
Code snippet:
```
V.dump(techniqueUsed='XGBoost')
```
## How to commit file, features and parameters to Vevesta
Vevesta is Feature and Technique Dictionary. The tool is free to use. Please create a login on [vevesta](https://www.vevesta.com/demo) . Then go to Setting section, download the access token. Place this token in the same folder as the jupyter notebook or python script. If by chance you face difficulties, please do mail vevestaX@vevesta.com.

You can commit the file(code),features and parameters to Vevesta by using the following command. You will find the project id for your project on the home page.

```
V.commit(techniqueUsed = "XGBoost", message="increased accuracy", version=1, projectId=1, repoName='My_Project')
```
A sample output excel file has been uploaded on google sheets. Its url is [here](https://docs.google.com/spreadsheets/d/11dzgjSumlEYyknQ2HZowVh0R1xvotJTqJR6WSqY7v3k/edit?usp=sharing)

## How to do EDA 
The library does EDA automatically on the data. 
```
V.EDA(df,Y=df["target"])

```

## How to check code into GitHub
In order to check-in the code to Git and Vevesta we would be requiring the two tokens mentioned below:

* Git-Access Token
* Vevesta Access Token
### How to Download the Git-Access Token ?

* Navigate to your Git account settings, then to Developer Settings. Click the Personal access tokens menu, then click Generate new token.

![https://catalyst.zoho.com/help/tutorials/githubbot/generate-access-token.html](https://miro.medium.com/max/1400/0*88zJhX1W7aAUzNvy.jpg)

* Select repo as the scope. The token will be applicable for all the specified actions in your repositories.

![https://catalyst.zoho.com/help/tutorials/githubbot/generate-access-token.html](https://miro.medium.com/max/1400/0*FJ-NBDehGsJAT3Ik.jpg)

* Click Generate Token: GitHub will display the personal access token. Make sure to copy the token and store it as a txt file by renaming it to git_token in the folder where the jupyter notebook or the python script is present.

![https://catalyst.zoho.com/help/tutorials/githubbot/generate-access-token.html](https://miro.medium.com/max/1400/0*KR6X8zv0n5o8VCCR.jpg)

We will use this token in the Integration function’s code, which will enable us to fetch the necessary information about the repositories from GitHub.

### How to Download the Vevesta Access Token ?

* Create a login on [vevesta](https://www.vevesta.com/signin).

![img](https://miro.medium.com/max/1400/1*RanYUoEr5wJNd2T1GBWI3Q.png)

* Then go to the Setting section, download the access token and place this token as it is without renaming in the same folder where the jupyter notebook or python script is present.

![img](https://miro.medium.com/max/1400/1*3lDbhhuZ53ePGGTGYLYLew.png)

This is how our folder looks when we have downloaded the above two tokens. Along with the two tokens we have with us: the jupyter notebook named ZIP.ipynb and Dataset named fish.csv.

![img](https://miro.medium.com/max/1400/1*gY8u1pGRGnMHuWS21xdt8w.png)


## Snapshots of output excel file
After running calling the dump or commit function for each run of the code. The features used, features engineered and the variables used in the experiments get logged into the excel file. In the below experiment, the commit/dump function is called 6 times and each time an experiment/code run is written into the excel sheet.
 
For example, code snippet used to track code runs/experiments are as below:

```
#import the vevesta Library
from vevestaX import vevesta as v
V=v.Experiment()
df = pd.read_csv("wine.csv") 
V.ds = df
df["salary_Ratio1"] = df["alchol_content"]/5
V.fe = df
epoch = 1000
accuracy = 90 #this will be a computed variable, may be an output of XGBoost algorithm
recall = 89  #this will be a computed variable, may be an output of XGBoost algorithm

```

For the above code snippet, each row in the excel sheet corresponds to an experiment/code run. The excel sheet will have the following:
1. Data Sourcing tab: Marks which Features (or columns) in wine.csv were read from the input file. Presence of the feature is marked as 1 and absence as 0.
2. Feature Engineering tab: Features engineered such as salary_Ratio1 exist as columns in the excel. Value 1 means that feature was engineered in that particular experiment and 0 means it was absent.
3. Modelling tab: This tab tracks all the variables used in the code. Say variable precision was computed in the experiment, then for the experiment ID i, precision will be a column whose value is computed precision variable. Note: V.start() and V.end() are code blocks that you might define. In that case, the code can have multiple code blocks. The variables in all these code blocks are tracked together. Let us define 3 code blocks in the code, first one with precision, 2nd one with recall and accuracy and 3rd one with epoch, seed and no of trees. Then for experiment Id <n>, all the variables, namely precision, recall, accuracy, epoch, seed and no. of trees will be tracked as one experiment and dumped in a single row with experiment id <n>. Note, if code blocks are not defined then it that case all the variables are logged in the excel file.
4. Messages tab: Data Scientists like to create new files when they change technique or approach to the problem. So everytime you run the code, it tracks the experiment ID with the name of the file which had the variables, features and features engineered.
5. EDA-correlation: correlation is calculated on the input data automatically. 
6. EDA-box Plot tab: Box plots for numeric features
7. EDA-Numeric Feature Distribution: Scatter plot with x axis as index in the data and y axis as the value of the data point.
8. EDA-Feature Histogram: Histogram of numeric features
 
Please note, EDA computation can be skipped by passing true during the creation of the object v.Experiment(True). The following is the code snippet:
```
#import the vevesta Library
from vevestaX import vevesta as v
V=v.Experiment(true)
```
### Sourced Data tab
![image](https://user-images.githubusercontent.com/81908188/169261190-f94c42d3-2ed7-4f33-b427-b1f4f8f9e4d2.png)

### Feature Engineering tab
![image](https://user-images.githubusercontent.com/81908188/169261608-b664936b-560f-421a-9b40-421d0a2e0400.png)

### Modelling tab
![image](https://user-images.githubusercontent.com/81908188/169261864-99528269-a816-4783-8354-18675bb21aff.png)
 
### Messages tab
![image](https://user-images.githubusercontent.com/81908188/169262663-d60edda6-1c6a-4236-a9d4-8ed665ec00e0.png)

### Experiments performance plots
![image](https://user-images.githubusercontent.com/81908188/160065501-c3b8a1c3-e75b-41fa-abea-cad3bb4b5add.png)
![image](https://user-images.githubusercontent.com/81908188/160065687-17821d15-2b12-4fc2-9978-f55d26c37ed0.png)


If you liked the library, please give us a github star and [retweet](https://twitter.com/vevesta1/status/1503747980188594178?s=20&t=3zXxSDS8WCddWcQHDxUrtg) .

For additional features, explore our tool at [Vevesta](https://vevesta.com) . For comments, suggestions and early access to the tool, reach out at vevestax@vevesta.com

We at Vevesta Labs are maintaining this library and we welcome feature requests. 

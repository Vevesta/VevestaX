# LIME
Data Science is a fast evolving field where most of the ML models are still treated as black boxes. Understanding the reason behind the predictions is one of the most important task one needs to perform in order to assess the trust if one plans to take action based on the predictions provided by the machine learning models.

This article deals with a novel explanation technique known as LIME that explains the predictions of any classifier in an interpretable and faithful manner.

## What is LIME?

LIME, or Local Interpretable Model-Agnostic Explanations, is an algorithm which explains the prediction of classifier or regressor by approximating it locally with an interpretable model. It modifies a single data sample by tweaking the feature values and observes the resulting impact on the output. It performs the role of an “explainer” to explain predictions from each data sample. The output of LIME is a set of explanations representing the contribution of each feature to a prediction for a single sample, which is a form of local interpretability.

## Why LIME?

LIME explains a prediction so that even the non-experts could compare and improve on an untrustworthy model through feature engineering. An ideal model explainer should contain the following desirable properties:

* Interpretable
LIME provides a qualitative understanding between the input variables and the response which makes it easy to understand.
* Local Fidelity
It might not be possible for an explanation to be completely faithful unless it is the complete description of the model itself. Having said that it should be at least locally faithful i.e. it must replicate the model’s behavior in the vicinity of the instance being predicted and here too LIME doesn’t disappoints us.
* Model Agnostic
LIME can explain any model without making any prior assumptions about the model.
* Global perspective
The LIME explains a representative set to the user so that the user can have a global intuition of the model.
Let’s have a quick look on a practical example of using LIME on a classification problem.

## Importing the libraries
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vevestaX import vevesta as v
Loading the Dataset
```

## Importing the dataset
```
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()
```
![img](https://miro.medium.com/max/1400/1*qdnv_-2fg3WE2NnaXyJWpw.png)
## Data Preprocessing and Train-Test-Split
```
x = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(x["Geography"],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

## Concatenate the Data Frames
x=pd.concat([x,geography,gender],axis=1)

## Drop Unnecessary columns
x=x.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
```
## Model Training
```
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)
```
## Introducing LIME
```
import lime
from lime import lime_tabular
interpretor = lime_tabular.LimeTabularExplainer(
    training_data=np.array(x_train),
    feature_names=x_train.columns,
    mode='classification'
)
exp = interpretor.explain_instance(
    data_row=x_test.iloc[5], ##new data
    predict_fn=classifier.predict_proba
)
exp.show_in_notebook(show_table=True)
```
This is how the explanations look for the index 5 of test data.

Note that LIME takes individual record as an input and then gives Explanation as the output.

![img](https://miro.medium.com/max/1400/1*Oq_Y6kjGtVWwk95d0S9Hew.png)

There are three parts to the explanation :

1. The left most section displays prediction probabilities, here in our case probability of being 0 comes out to be 0.33 whereas 0.67 for 1.
2. The middle section returns the most important features. For the binary classification task, it would be in 2 colors orange/blue. Attributes in orange support class 1 and those in blue support class 0. Age >44.00 supports class 1. Float point numbers on the horizontal bars represent the relative importance of these features.
3. The color-coding is consistent across sections. It contains the actual values of the variables.

## Dumping the Experiment
```
V.dump(techniqueUsed='LIME',filename="LIME.xlsx",message="LIME was used",version=1)
```
## Brief Intro about VevestaX
VevestaX is an open source Python package which includes a variety of features that makes the work of a Data Scientist pretty much easier especially when it comes to analyses and getting the insights from the data.

The package can be used to extract the features from the datasets and can track all the variables used in code.

The best part of this package is about its output. The output file of the VevestaX provides us with numerous EDA tools like histograms, performance plots, correlation matrix and much more without writing the actual code for each of them separately.

## How to Use VevestaX?

* Install the package using:
```
pip install vevestaX
```
* Import the library in your kernel as:
```
from vevestaX import vevesta as v
V=v.Experiment()
```
* To track the feature used:
```
V.ds = dataframe
```
* To track features engineered
```
V.fe = dataframe
```
* Finally in order to dump the features and variables used into an excel file and to see the insights what the data carries use:
```
V.dump(techniqueUsed='LIME',filename="LIME.xlsx",message="AIF 360 was used",version=1)
```
Following are the insights we received after dumping the experiment:

![img](https://miro.medium.com/max/1400/1*rqwquzYt6cLnN290slxDcw.png)

![img](https://miro.medium.com/max/1400/1*zmaU_45TtYcWIOMzUt2B-Q.png)

![img](https://miro.medium.com/max/1400/1*jtNb5_W4kZY43wR3FFfczQ.png)

![img](https://miro.medium.com/max/1400/1*_E7I5exg3CRL0ZDy9BjD5g.png)

![img](https://miro.medium.com/max/1400/1*bje5VZxZkqeCzim985v3dw.png)

![img](https://miro.medium.com/max/1400/1*Wvyb3HekjE959Xt3kUHlKg.png)

![img](https://miro.medium.com/max/1400/1*ldMMnvKqP_lDDbWeol8t3A.png)

![img](https://miro.medium.com/max/1400/1*3Sa9CLVbQ1chkjQBeYsf_Q.png)

Here ends our look at using the LIME Package in Machine Learning Models.

For Source Code [Click Here](https://gist.github.com/sarthakkedia123/7f305ade7478779838f844e3b787011d#file-lime-ipynb)

## References

* [Towards DataScience](https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5)
* [Papers with Code](https://paperswithcode.com/method/lime)
* [VevestaX GitHub Link](https://github.com/Vevesta/VevestaX)
* [Original LIME Tabular Tutorial](https://www.vevesta.com/blog/8_Using_LIME_to_understand_NLP_Models?utm_source=Github_VevestaX_LIME_Tabular)

## Credits
[Vevesta](https://www.vevesta.com?utm_source=Github_VevestaX_LIME_Tabular) is Your Machine Learning Team's Collective Wiki: Save and Share your features and techniques. Explore [Vevesta](https://www.vevesta.com?utm_source=Github_VevestaX_LIME_Tabular) for free. For more such stories, follow us on twitter at [@vevesta1](http://twitter.com/vevesta1).

## Author 
Sarthak Kedia

# VevestaX

![image](https://user-images.githubusercontent.com/81908188/142753559-9f94639b-324b-4734-a183-cd7d2c97a3fc.png)

# Track failed and successful Machine Learning experiments as well as features.

VevestaX is an open source Python package for ML Engineers and Data Scientists.  It includes modules for tracking features sourced from data, feature engineering and variables. The output is an excel file which has tabs namely, data sourcing, feature engineering and modelling. The library can be used with Jupyter notebook, IDEs like spyder and while running the python script through command line.



How to install the library:

$ pip install vevestaX

How to import a library and create the object
![image](https://user-images.githubusercontent.com/81908188/140261967-6cf57c32-d58c-4f85-8eba-7a5387295fa1.png)

How to extract features present in input data.
![image](https://user-images.githubusercontent.com/81908188/141691820-511ebba1-bc5a-4ce4-acd0-cd23ae3cd782.png)

How to extract engineered features
![image](https://user-images.githubusercontent.com/81908188/140041279-7ecd6444-a9ba-4e87-a0e5-46435c759d18.png)

How to track variables used in modelling section of the code. V.start() and V.end() form a code block and can be called multiple times in the code to track variables used within the code block. Any technique such as XGBoost, decision tree, etc can be used within this code block.
![image](https://user-images.githubusercontent.com/81908188/140041422-97be7287-111d-40c3-bc8f-d921db90acf8.png)

How to dump the features and modelling variables in an given xlsx file
![image](https://user-images.githubusercontent.com/81908188/140653881-1698d7ba-1c0f-4879-8a96-a90123108165.png)

Alternatively, write the experiment into the default file, vevesta.xlsx
![image](https://user-images.githubusercontent.com/81908188/140653897-6654e94b-a332-49a2-a7b7-416cb5bded5c.png)


A sample output excel file has been uploaded on google sheets. Its url is https://docs.google.com/spreadsheets/d/1NXHqzmGegyHm2TnvGFpoe3MSI8N5YAfx/edit?usp=sharing&ouid=103382336064969333270&rtpof=true&sd=true



For additional features, explore our tool at www.vevesta.com . For comments, suggestions and early access to the tool, reach out at vevestax@vevesta.com

We at vevesta Labs are maintaining this library and we welcome feature requests. Find detailed blog on the vevestaX on https://medium.com/@priyanka_60446/vevestax-open-source-library-to-track-failed-and-successful-machine-learning-experiments-and-data-8deb76254b9c

PS: This library doesn't send back data to our servers.

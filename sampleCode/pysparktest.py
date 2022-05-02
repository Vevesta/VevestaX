from vevestaX import vevesta
from pyspark.sql import SparkSession
# import pandas as pd

import os
import sys

#
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

V = vevesta.Experiment()


spark = SparkSession.builder.appName("vevesta").getOrCreate()
df_pyspark = spark.read.format("csv").option("header", "true").load("data.csv")
# df_pyspark = spark.read.csv('file:///C:/Users/physc/MachineLearning/vevesta-Model/sampleCode/wine.csv')
# df_pyspark = pd.read_csv("data.csv")

sc = spark.sparkContext
sc.setLogLevel("OFF")

V.dataSourcing = df_pyspark
print(V.ds)

# Do some feature engineering
# df_pyspark["salary_feature"]= df_pyspark["Salary"] * 100/ df_pyspark["House_Price"]
# df_pyspark['salary_ratio1']=df_pyspark["Salary"] * 100 / df_pyspark["Months_Count"] * 100


# performing column operation on pyspark dataframe
# df_pyspark = df_pyspark.withColumn('Ash_feature', df_pyspark.Ash*100)
# df_pyspark = df_pyspark.withColumn('salary_ratio1', df_pyspark.Salary*100)
# print(df_pyspark.show())

#Extract features engineered
V.fe=df_pyspark

#Print the features engineered
print(V.fe)

V.dump(techniqueUsed='XGBoost', filename="../vevestaX/vevestaDump.xlsx", message="precision is tracked", version=1)


# Classification with K-Means and Vevestax library.

In this article we will be focusing on a very well-known unsupervised machine learning technique ‘K-Means’ and will be using a very efficient python package known as ‘VevestaX’ in order to perform Exploratory Data Analysis and Experiment Tracking.

## Table of Contents
1.  [K-Mean Clustering](https://github.com/Vevesta/VevestaX/blob/main/tutorials/Kmeans/tutorial_kmeans.md#k-mean-clustering)
2.  [How the K-means algorithm works?](https://github.com/Vevesta/VevestaX/blob/main/tutorials/Kmeans/tutorial_kmeans.md#how-the-k-means-algorithm-works)
3.  [VevestaX](https://github.com/Vevesta/VevestaX/blob/main/tutorials/Kmeans/tutorial_kmeans.md#vevestax)
4.  [How To Use VevestaX?](https://github.com/Vevesta/VevestaX/blob/main/tutorials/Kmeans/tutorial_kmeans.md#how-to-use-vevestax)
5.  [How to perform clustering using K Means and VevestaX?](https://github.com/Vevesta/VevestaX/blob/main/tutorials/Kmeans/tutorial_kmeans.md#how-to-perform-clustering-using-k-means-and-vevestax)
6.  [References](https://github.com/Vevesta/VevestaX/blob/main/tutorials/Kmeans/tutorial_kmeans.md#references)

## K-Mean Clustering
K-means clustering is one of the easiest and most popular unsupervised machine learning algorithms. Clustering algorithms are used to cluster similar data points, each based on their own definition of similarity. The K-means algorithm identifies the number of clusters, k and then assigns each data point to the nearest cluster.

## How the K-means algorithm works?
While learning from the data, the K-means algorithm starts with a first group of randomly selected centroids. These centroids are used as the initial points assigned to every cluster. K-means performs iterative (repetitive) calculations to optimize the positions of the centroids. It does this by minimizing the distance of points from the centroid.

It stops creating and optimizing the clusters when either:

* There is no change in the values of the centroid because the clustering has been successful.
* The defined number of iterations has been achieved.

## VevestaX
VevestaX is an open source Python package which includes a variety of features that makes the work of a Data Scientist pretty much easier especially when it comes to analyses and getting the insights from the data.

The package can be used to extract the features from the datasets and can track all the variables used in code.

The best part of this package is about its output. The output file of the VevestaX provides us with numerous EDA tools like histograms, performance plots, correlation matrix and much more without writing the actual code for each of them separately.

## How To Use VevestaX?
Install the package as follows:

```
pip install vevestaX
```

Import and create a vevesta object as follows:

```
from vevestaX import vevesta as v
V=v.Experiment()
```

To track the feature used:

```
V.ds = df
```

where df is the pandas dataframe with the input features

![Data Sourcing](https://miro.medium.com/max/875/1*k-xRg908ebCeGNjwpSVtSA.png)

To track features engineered

```
V.fe = df
```

Finally in order to dump the features and variables used into an excel file and to see the insights what the data carries use:

```
V.dump(techniqueUsed=’Model_Name’,filename=”vevestaDump.xlsx”,message=”precision is tracked”,version=1)
```


Following are the insights we received after dumping the features:

![Data Sourcing](https://miro.medium.com/max/875/1*Cq4g-mxTeIEXO7ENiYlIBQ.png)

![Modeling](https://miro.medium.com/max/875/1*YogmO8lQO0-a9zQlxgyDLw.png)

![Messages](https://miro.medium.com/max/875/1*imcZJfOIAPtNBMYRwsd76A.png)

![EDA-correlation](https://miro.medium.com/max/875/1*w8WMUlJ42c4ep3HwhMWJRw.png)

![EDA-scatterplot](https://miro.medium.com/max/875/1*FJYB3ypjtXagUWIVUY4n8g.png)

## How to perform clustering using K Means and VevestaX?

So what we have basically done is, firstly we have imported the necessary libraries and loaded the dataset.

![image](https://miro.medium.com/max/875/1*cppo5SPNWFe2YtSJYbkMqA.png)

Thereafter we had performed the train_test _split in order to get the train and test dataset.

![image](https://miro.medium.com/max/875/1*800Qu7ojsH8EwIxkRqKHkg.png)

Next, we cluster the data using K-Means. The number of clusters to be formed will be same as the classes in the data. The K-Means model is fitted on the train data and then the labels for test data are predicted. Finally, we calculate the baseline NMI score for the model.

![image](https://miro.medium.com/max/875/1*DLWTnU0X4nDhrtxMfCnd_Q.png)

Next in order to get the centroids of the clusters we used:

``` 
model_kmeans.cluster_centers_
```


![](https://miro.medium.com/max/875/1*xc8WsQQJdFWDEaKECC1qNw.png)

Finally we have dumped the data into Excel File using VevestaX.

![](https://miro.medium.com/max/875/1*eXPHzj5ckS1eWemOp-a9hw.png)

[*For Source Code Click Here*](https://gist.github.com/sarthakkedia123/bd77515160a0b2d953266e0302268fd2)

## References

1.  [FRUFS’s Github](https://github.com/atif-hassan/FRUFS)

2.  [FRUFS Author’s article](https://www.deepwizai.com/projects/how-to-perform-unsupervised-feature-selection-using-supervised-algorithms)

3.  [FRUFS article](https://www.vevesta.com/blog/1)

4.  [VevestaX article](https://medium.com/@priyanka_60446/vevestax-open-source-library-to-track-failed-and-successful-machine-learning-experiments-and-data-8deb76254b9c)

5.  [VevestaX GitHub Link](https://github.com/Vevesta/VevestaX)

Vevesta is the next generation Portfolio for Machine Learning Project : Save and share machine learning projects. Explore [*Vevesta*](https://www.vevesta.com/?utm_source=vevestax_github) for free. For more such stories, follow us on twitter at [*@vevesta1*](http://twitter.com/vevesta1).

**Author**

Sarthak Kedia

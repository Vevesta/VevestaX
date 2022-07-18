## DBSCAN Clustering
Clustering is the technique of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters.

It is an unsupervised learning method so there is no label associated with data points. The algorithm tries to find the underlying structure of the data. It comprises of many different methods, few of which are: K-Means (distance between points), Affinity propagation (graph distance), Mean-shift (distance between points), DBSCAN (distance between nearest points), Gaussian mixtures, etc.

In this article we will be focusing on the detailed study of DB Scan Algorithm, so let’s begin.

Partition-based or hierarchical clustering techniques are highly efficient with normal shaped clusters. However, when it comes to arbitrary shaped clusters or detecting outliers, density-based techniques are more efficient.

Lets consider the following figures..

![img](https://miro.medium.com/max/1024/1*iSyysRBup5mfm3U9s4uUQA.png)
![img](https://miro.medium.com/max/812/1*J-h6Bt9xWPfLswS66ijEvw.png)
![img](https://miro.medium.com/max/672/1*oGM9TBm_Cth-bs06rvjAmw.png)
The above images are taken from the [article](https://towardsdatascience.com/dbscan-clustering-explained-97556a2ad556) published in Towards Data Science.


The data points in these figures are grouped in arbitrary shapes or include outliers. Thus Density-based clustering algorithms are very efficient in finding high-density regions and outliers when compared with Normal K-Means or Hierarchical Clustering Algorithms.

**DBSCAN**

The DBSCAN algorithm stands for Density-Based Spatial Clustering of Applications with Noise. It is capable to find arbitrary shaped clusters and clusters with noise (i.e. outliers).

The main idea behind the DBSCAN Algorithm is that a point belongs to a cluster if it is close to several points from that cluster.

There are two key parameters of DBSCAN:

* **eps(Epsilon)**: The distance that defines the neighborhoods. Two points are considered to be neighbors if the distance between them is less than or equal to eps.
* **minPts(minPoints)**: Minimum number of data points that are required to define a cluster.
Based on these two parameters, points are classified as core point, border point, or outlier:

* **Core point:** A point is said to be a core point if there are at least minPts number of points (including the point itself) in its surrounding area with radius eps.
* **Border point:** A point is a border point if it is reachable from a core point and there are less than minPts number of points within its surrounding area.
* **Outlier:** A point is an outlier if it is not a core point and not reachable from any core points.
The following figure has eps=1 and minPts=5 and is taken from [researchgate.net](https://www.researchgate.net/publication/334809161_ANOMALOUS_ACTIVITY_DETECTION_FROM_DAILY_SOCIAL_MEDIA_USER_MOBILITY_DATA).

![img](https://miro.medium.com/max/704/1*FnV-x4Kpo7oBwUnTWwaccA.png)

**How does the DBSCAN Algorithm create Clusters?**

The DBSCAN algorithm starts by picking a point(one record) x from the dataset at random and assign it to a cluster 1. Then it counts how many points are located within the ε (epsilon) distance from x. If this quantity is greater than or equal to minPoints (n), then considers it as core point, then it will pull out all these ε-neighbors to the same cluster 1. It will then examine each member of cluster 1 and find their respective ε -neighbors. If some member of cluster 1 has n or more ε-neighbors, it will expand cluster 1 by putting those ε-neighbors to the cluster. It will continue expanding cluster 1 until there are no more data points to put in it.

In the latter case, it will pick another point from the dataset not belonging to any cluster and put it to cluster 2. It will continue like this until all data points either belong to some cluster or are marked as outliers.

**DBSCAN Parameter Selection**

DBSCAN is extremely sensitive to the values of epsilon and minPoints. A slight variation in these values can significantly change the results produced by the DBSCAN algorithm. Therefore, it is important to understand how to select the values of epsilon and minPoints.

* **minPoints(n):**
As a starting point, a minimum n can be derived from the number of dimensions D in the data set, as n ≥ D + 1. For data sets with noise, larger values are usually better and will yield more significant clusters. Hence, n = 2·D can be a suggested valued, however this is not a hard and fast rule and should be checked for multiple values of n.

* **Epsilon(ε):**
If a small epsilon is chosen, a large part of the data will not be clustered whereas, for a too high value of ε, clusters will merge and the majority of objects will be in the same cluster. Hence, the value for ε can then be chosen by using a [k-graph](https://en.wikipedia.org/wiki/Nearest_neighbor_graph). Good values of ε are where this plot shows an “elbow”.

**Code**

*Importing the Libraries*
```
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
%matplotlib inline
```
*Generating Clustered Data From Sklearn*
```
X, y = make_blobs(n_samples=1000,cluster_std=0.5, random_state=0)
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
```
![img](https://miro.medium.com/max/948/1*U7SlHSmNUUB9p_sNUPW5Tg.png)

*Initialization and Fitting the model.*
```
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.4, min_samples=20)
db.fit(X)
y_pred = db.fit_predict(X)
```

*Plotting the clustered data points*
```
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1],c=y_pred)
plt.title("Clusters determined by DBSCAN")
plt.show()
```
![img](https://miro.medium.com/max/948/1*tYEfdtBbIFf7KzIsjmPfsw.png)

The clusters in this sample dataset do not have arbitrary shapes but here we see that DBSCAN performed really good at detecting outliers which would not be easy with partition-based (e.g. k-means) or hierarchical (e.g. agglomerative) clustering techniques. If we would have applied DBSCAN to a dataset with arbitrary shaped clusters, DBSCAN would still outperform the rest of the two clustering techniques mentioned above.

**References**

* [MyGreatLearning](https://www.mygreatlearning.com/blog/dbscan-algorithm/)
* [Soner Yıldırım](https://towardsdatascience.com/dbscan-clustering-explained-97556a2ad556)
* [DBscan Original article](https://www.vevesta.com/blog/11_DBSCAN_Clustering?utm_source=GitHub_VevestaX_DBScan)

## Credits
[Vevesta](https://www.vevesta.com?utm_source=GitHub_VevestaX_DBScan) is Your Machine Learning Team's Collective Wiki: Save and Share your features and techniques. Explore [Vevesta](https://www.vevesta.com?utm_source=GitHub_VevestaX_DBScan) for free. For more such stories, follow us on twitter at [@vevesta1](http://twitter.com/vevesta1).

**Author:** Sarthak Kedia

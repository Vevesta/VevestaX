
# Quick overview of methods used to handle overfitting in Shallow and Deep Neural Network

Overfitting is due to the fact that the model is complex, only memorizes the training data with limited generalizability and cannot correctly recognize different unseen data.

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https://bucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com/public/images/77231416-6824-4514-b177-490b98270e10_877x747.png)

According to [authors](https://link.springer.com/article/10.1007/s10462-021-09975-1), reasons for overfitting are as follows:

1. Noise of the training samples,
2. Lack of training samples (under-sampled training data),
3. Biased or disproportionate training samples,
4. Non-negligible variance of the estimation errors,
5. Multiple patterns with different non-linearity levels that need different learning models,
6. Biased predictions using different selections of variables,
7. Stopping training procedure before convergence or dropping in a local minimum,
8. Different distributions for training and testing samples.

## Methods to handle overfitting:

1. Passive Schemes: Methods meant to search for suitable configuration of the model/network and are some times called as Model selection techniques or hyper-parameter optimization techniques.
2. Active Schemes: Also, referred as regularization techniques, this method introduces dynamic noise during model training time.
3. Semi - Active Schemes: In this methodology, the network is changed during the training time. The same is achieved either by network pruning during training or addition of hidden units during the training.

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https://bucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com/public/images/8d03b67b-2c7e-4a47-a713-1468d8091773_741x736.png)

## References:

1. [A systematic review on overftting control in shallow and deep neural networks](https://link.springer.com/article/10.1007/s10462-021-09975-1)
2. [Overfitting in Shallow and Deep Neural Network on Vevesta](https://www.vevesta.com/blog/25-Handling-overfitting-in-Shallow-and-Deep-Neural-Network)
3. [Overfitting in Shallow and Deep Neural Network on Substack](https://vevesta.substack.com/p/deep-dive-in-causes-of-overfitting)

## Credits

The above article is sponsored by [vevesta](https://www.vevesta.com/).

[Vevesta](https://www.vevesta.com/): Your Machine Learning Team’s Feature and Technique Dictionary: Accelerate your Machine learning project by using features, techniques and projects used by your peers. Explore [Vevesta](https://www.vevesta.com/) for free. For more such stories, follow us on twitter at [@vevesta_labs](https://twitter.com/vevesta_labs).
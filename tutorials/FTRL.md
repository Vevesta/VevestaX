
# A look into little known but powerful optimizer by Google, FTRL

### A look into little known but powerful optimizer by Google, FTRL

When training a neural network, its weights are initially initialized randomly and then they are updated in each epoch in a manner such that they reduce the overall loss of the network. In each epoch, the output of the training data is compared to actual data with the help of the loss function to calculate the error and then the weight is updated accordingly. But how do we know how to update the weight such that it reduce the loss?

This is essentially an optimization problem where the goal is to optimize the loss function and arrive at ideal weights. The method used for optimization is known as Optimizer.

Optimizers are techniques or algorithms used to decrease loss (an error) by tuning various parameters and weights, hence minimizing the loss function, providing better accuracy of model faster.

Follow The Regularized Leader (FTRL) is an optimization algorithm developed at Google for click-through rate prediction in the early 2010s. It is best suited for shallow models having sparse and large feature spaces. The algorithm is described by [McMahan et al., 2013.](https://research.google.com/pubs/archive/41159.pdf) This version supports both shrinkage-type L2 regularization (summation of L2 penalty and loss function) and online L2 regularization.

The Ftrl-proximal algorithm, abbreviated for Follow-the-regularized-leader (FTRL) can give a good performance vs. sparsity tradeoff.

Ftrl-proximal uses its own global base learning rate and can behave like Adagrad with learning_rate_power=-0.5, or like gradient descent with learning_rate_power=0.0.

```
tf.keras.optimizers.Ftrl(
    learning_rate=0.001,
    learning_rate_power=-0.5,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    name="Ftrl",
    l2_shrinkage_regularization_strength=0.0,
    beta=0.0,
    **kwargs
)
```

## Initialization

```
n = 0
sigma = 0
z = 0
```

#### Notation

* lr is the learning rate
* g is the gradient for the variable
* lambda_1 is the L1 regularization strength
* lambda_2 is the L2 regularization strength

#### Update rule for one variable w

```
prev_n = n
n = n + g ** 2
sigma = (sqrt(n) - sqrt(prev_n)) / lr
z = z + g - sigma * w
if abs(z) < lambda_1:
  w = 0
else:
  w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / alpha + lambda_2)
```

#### Arguments

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F28dd1db4-01b1-4b93-a964-4dc420d168f1_1000x631.png)

## Uses of FTRL optimizer

#### 1. Ranking Documents
Ranking means sorting documents by relevance to find contents of interest with respect to a query. Ranking models typically work by predicting a relevance score s = f(x) for each input x = (q, d) where q is a query and d is a document. Once we have the relevance of each document, we can sort (i.e. rank) the documents according to those scores.

#### 2. Multi-Armed Bandit (MAB) problem
In this problem, as mentioned in [Towards an Optimization Perspective for Bandits Problem](https://cseweb.ucsd.edu/classes/wi22/cse203B-a/proj22/17.pdf), a decision-maker is faced with a fixed arm set and needs to design a strategy to pull an arm to minimize the cumulative loss, termed regret. At each round, the decision-maker adapts the pulling strategy by solving an optimization problem (OP), and the solution of OP is a probability distribution over all arms. FTRL-based algorithm is one of the methods achieve the best of both worlds, i.e., stochastic and adversarial setting, in bandit problem with graph feedback.

#### 3. Document Retrieval, Recommendation Systems, and Disease Diagnosis
[A mini-batch stochastic gradient method for sparse learning to rank](http://www.ijicic.org/ijicic-140403.pdf) states that the algorithm for rank learning begins with formulating sparse learning to rank as a mini-batch based convex optimization problem with L1 regularization. Then for the problem that simple adding L1 term does not necessarily induce the sparsity, FTRL method is adopted for inner optimization, which can obtain good solution with high sparsity.

#### 4. Online Advertising
The underlying driving technology for online advertising is Click-Through Rates (CTR) estimation, in which the task is to predict the click probability of the browsers for some commodities in certain scenarios. Accurate prediction of CTR will not only benefit advertisers’ promotion of products but also ensure users’ good experiences and interests. The FTRL model combined the power of forward backward splitting algorithm (FOBOS) and regularized dual averaging algorithm (RDA) and has been successfully be used for online optimization of logistic regression model. It uses both L1 and L2 regularization terms in the iterative process, which greatly improves the prediction of the model. This was deduced by [A New Click-Through Rates Prediction Model Based on Deep & Cross Network.](https://www.mdpi.com/1999-4893/13/12/342/htm)

#### 5. High-Dimensional Sparse Streaming Data Analysis
An algorithm based on FTRL, FTRL-AUC, as proposed by [Online AUC Optimization for Sparse High-Dimensional Datasets](https://arxiv.org/pdf/2009.10867.pdf). can process data in an online fashion with a much cheaper per-iteration cost O(k), making it amenable for high-dimensional sparse streaming data analysis. It significantly improves both run time and model sparsity while achieving competitive Area Under the ROC Curve (AUC) scores compared with the state-of-the-art methods. Comparison with the online learning method for logistic loss demonstrates that FTRL-AUC achieves higher AUC scores especially when datasets are imbalanced.

## Advantages
1. Can minimize loss function better.

## Disadvantages
1. Cannot achieve adequate stability if the range of the regulariser is insufficient.
2. If the range of the regulariser is huge, then it’s far away from the optimal decision.

## References:

1. [Ftrl (keras.io)](https://keras.io/api/optimizers/ftrl/)
2. [tf.keras.optimizers.Ftrl | TensorFlow v2.9.1](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl)
3. [Optimizers in Tensorflow-GeeksforGeeks](https://www.geeksforgeeks.org/optimizers-in-tensorflow/)
4. [McMahan et al., 2013.](https://research.google.com/pubs/archive/41159.pdf)
5. [Towards an Optimization Perspective for Bandits Problem.](https://cseweb.ucsd.edu/classes/wi22/cse203B-a/proj22/17.pdf)
6. [A mini-batch stochastic gradient method for sparse learning to rank.](http://www.ijicic.org/ijicic-140403.pdf)
7. [A New Click-Through Rates Prediction Model Based on Deep & Cross Network.](https://www.mdpi.com/1999-4893/13/12/342/htm)
8. [Online AUC Optimization for Sparse High-Dimensional Datasets.](https://www.vevesta.com/blog/Online%20AUC%20Optimization%20for%20Sparse%20High-Dimensional%20Datasets.)
9. [FTRL article on Vevesta](https://www.vevesta.com/blog/23-FTRL)
10. [FTRL article on Substack](https://vevesta.substack.com/p/a-look-into-little-known-but-powerful)
## Credits

The above article is sponsored by [vevesta](https://www.vevesta.com/).

[Vevesta](https://www.vevesta.com/): Your Machine Learning Team’s Feature and Technique Dictionary: Accelerate your Machine learning project by using features, techniques and projects used by your peers. Explore [Vevesta](https://www.vevesta.com/) for free. For more such stories, follow us on twitter at [@vevesta_labs](https://twitter.com/vevesta_labs).
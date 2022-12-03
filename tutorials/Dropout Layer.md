
# Uncovering Hidden Insights into Dropout Layer

Problem faced in training deep neural networks

Deep neural networks with a large number of parameters suffer from overfitting and are also slow to use.


## What is Dropout, in short?

According to the authors, the key idea of dropout is based on randomly droping units (along with their connections) from the neural network during training. This stops units from “co-adapting too much”. During training, exponential number of different “thinned” networks are sampled. “At test time, it is easy to approximate the effect of averaging the predictions of all these thinned networks by simply using a single unthinned network that has smaller weights. This significantly reduces overfitting and gives major improvements over other regularization methods”. Also, dropout layer avoids co-adaptation of neurons by making it impossible for two subsequent neurons to rely solely on each other.

Using dropout can be viewed as training a huge number of neural networks with shared parameters and applying bagging at test time for better generalization.

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F4f2af7a2-7f48-421b-a697-13d5858f6014_950x465.png)

## Deep Dive into Dropout

The term “dropout” refers to dropping out units, both hidden and visible, in a neural network. Dropping a neuron/unit out means temporarily removing it from the network along with all its incoming and outgoing connections. The choice of which units to drop is random. In the simplest case, each unit is retained with a fixed probability p independent of other units, where p can be chosen using a validation set.

### Things to keep in mind while using Dropout in experiments

1. For a wide variety of networks and tasks, dropout should be set to 0.5.
2. For input units, the optimal dropout is usually closer to 0 than 0.5, or alternatively, optimal probability of retention should be closer to 1.
3. Note that p is 1- (dropout probability) and dropout probability is what we set in neural network while coding in keras or Tensorflow.
4. While training network with SGD, dropout layer along with maxnorm regularization, large decaying learning rates and high momentum provides a significant boost over just using dropout.

### How does Dropout work?

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F6e2f7580-6b89-447b-8996-66cfbcbee955_898x373.png)

As shown in Figure 2, during training, the unit/neuron is present with probability p and is connected with units in the next layer with weights, w. During testing phase, the unit is always present and its weights are multiplied by p.

Dropout is applied to the neural network of n units and 2^n possible thinned neural networks are generated. A thinned neural network is a neural network which has dropped some units and their corresponding connections, as shown in figure 1b. The interesting part is that despite thinning, these networks all share weights so that the total number of parameters is still O(n^2), or less. During training phase for each data point, a certain permutation of units are switched off and a new thinned network is sampled and trained. Each thinned network gets trained very rarely, if at all.

At test time, since it’s not feasible to explicitly average the predictions from exponentially many thinned models. The idea is that the full neural net is used at test time without dropout. Inorder to compensate for dropout being applied during training phase, if a neuron is retained with probability p during training, the outgoing weights of that neuron are multiplied by p at test time, as can be seen in Figure 2. By using this methodology, during the testing phase, 2^n networks with shared weights can be combined into a single neural network. According to [authors](http://cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf), it was noticed using this methodology leads to significantly lower generalization error on a wide variety of classification problems compared to training with other regularization methods.

## Applications of Dropout layer

Few examples of domains where dropout is finding extensive use are as follows:

1. According to Merity et al., dropout is the norm for NLP problems as it is much more effective than methods such as L2 regularization
2. In vision, dropout is often used to train extremely large models such as EfficientNet-B7.

## References:

1. [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182)
2. [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf)
3. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
4. [Dropout as data augmentation](https://arxiv.org/pdf/1506.08700.pdf)
5. [Dropout Layer Article on Substack](https://vevesta.substack.com/p/uncovering-hidden-insights-into-dropout)
6. [Dropout Layer Article on Vevesta](https://www.vevesta.com/blog/21-Dropout-Layer)

## Credits:

The above article is sponsored by [vevesta.](https://www.vevesta.com/)

[Vevesta](https://www.vevesta.com/): Your Machine Learning Team’s Feature and Technique Dictionary: Accelerate your Machine learning project by using features, techniques and projects used by your peers. Explore [Vevesta](https://www.vevesta.com/) for free. For more such stories, follow us on twitter at [@vevesta_labs](https://twitter.com/vevesta_labs).
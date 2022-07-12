# Why early stopping of the neural network might not always be the optimal decision?
We all have practically encountered that while training a neural network, after a limited number of steps the loss function begins to slow down significantly. Then after resting for a long period of time, the loss may suddenly start dropping rapidly again for no explanatory cause, and this process continues till we run out of steps.

![img](https://cdn-images-1.medium.com/max/900/0*rA05n6siCddLinjn.png)
[Image Credits](https://cdn-images-1.medium.com/max/900/0*rA05n6siCddLinjn.png).

As we can notice from figure (a), for the first 10 epochs, the loss decreases drastically but thereafter, it tends to remain constant for a long period of time. After having such a long computation, the loss again tends to fall drastically as shown in figure (b), and then again becomes almost constant.

Many of us might make our decision on this model by observing the curve shown in fig (a), but the thing is that if we train our network for more epochs then there is a chance of convergence of the model at a better point.

These plateaus complicates our decision on where to stop the gradient drop and also slow down the convergence as in order to traverse a plateau thinking that there might be a a possible chance of decreasing the loss more number of iterations are required.

## Cause of Plateau
There are two main causes due to which the formation of plateau takes place and they are as follows:
* Saddle Point
* Local Minima

![img](https://cdn-images-1.medium.com/max/900/1*-ya2AEsB91XDsjXkMjs-tg.png)
[Image Credits](https://medium.com/r/?url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FDefinition-of-grey-level-blobs-from-local-minima-and-saddle-points-2D-case_fig1_10651758).

## Saddle Point

![img](https://cdn-images-1.medium.com/max/900/0*OQE_bSxccQ6R45P5.png)
[Image Credits](https://medium.com/r/?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSaddle_point)

The major problem with the saddle points is that at the saddle point of a particular function the gradient is zero and that does not represent the maximum and minimum value. The machine learning algorithm and optimization algorithm in a neural network is being optimized by the status of the gradient and if the gradient is zero the model gets stuck.

## Local Minima

![img](https://cdn-images-1.medium.com/max/900/0*UfvC_Z1JespJIOcr.png)
[Image Credits](https://www.researchgate.net/figure/1st-order-saddle-point-in-the-3-dimensional-surface-Surface-is-described-by-the_fig7_280804948)

The point, in this case, is an extremum, which is excellent, but the gradient is zero. If our learning rate is too low, we may not be able to escape the local minimum. As noticed earlier in fig(a), the loss value in our hypothetical training situation began balancing around some constant number, a  prominent reason behind this is the formation of these kinds of local minimum

## Effect of Learning Rate
The learning rate hyperparameter determines how quickly the model learns. While an increased learning rate allows the model to learn faster, it may result in a less-than-optimal final set of weights. However, a slower learning rate may allow the model to acquire a more optimal, or perhaps a globally optimal set of weights, but it will take much longer to train. The problem with a low learning rate is that it may never converge or it may get trapped on a suboptimal solution.

Thus, learning rate plays a crucial role in order to overcome the plateau phenomenon, techniques like scheduling the learning rate or cyclical learning rate are used for the same.

## Methods to Overcome a Plateau Problem
Following are the approaches which might be used to tweak the learning rates in order to overcome the plateau problem:

## Scheduling the Learning Rate
Scheduling the learning rate is the most common approach, which proposes starting with a reasonably high learning rate and gradually reducing it during training. The whole idea behind this is that we want to get from the initial parameters to a range of excellent parameters values quickly, but we also want a learning rate low enough to explore the deeper, but narrower regions of the loss function.

![img](https://cdn-images-1.medium.com/max/900/0*lv38Hvzb6PwX0ZNt.png)
[Image Credits](https://medium.com/r/?url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FStep-Decay-Learning-Rate_fig3_337159046)

An example of this is Step decay in which the learning rate is lowered by a certain percentage after a certain number of training epochs.

## Cyclical Learning Rate
Leslie Smith provided a cyclical learning rate scheduling approach with two bound values that fluctuate.

Cyclical learning scheme displays a sublime balance between passing over local minima while still allowing us to look around in detail.

![img](https://cdn-images-1.medium.com/max/900/0*lgTFEwR5GT2u2EX4.png)

[Image Credits](https://medium.com/r/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F1506.01186.pdf)

Thus Scheduling the learning rate helps us in order to overcome the plateau problems faced during optimizing the neural networks.

## References:
* [Analytics Vidhya](https://medium.com/r/?url=https%3A%2F%2Fanalyticsindiamag.com%2Fwhat-is-the-plateau-problem-in-neural-networks-and-how-to-fix-it%2F)
* [Plateau Phenomenon by Mark Ainsworth](https://medium.com/r/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F2007.07213.pdf)
* [Cyclical Learning Rate](https://medium.com/r/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F1506.01186.pdf)
* [Find best learning rate on Plateau](https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2FJonnoFTW%2Fkeras_find_lr_on_plateau)

## Author
Sarthak Kedia
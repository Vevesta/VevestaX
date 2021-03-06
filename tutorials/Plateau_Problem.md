# Pitfalls of early stopping neural network
We've all noticed that after a certain number of training steps, the loss starts to slow significantly. After a long period of steady loss, the loss may abruptly resume dropping rapidly for no apparent reason, and this process will continue until we run out of steps.

![img](https://cdn-images-1.medium.com/max/900/0*rA05n6siCddLinjn.png)
[Image Credits](https://cdn-images-1.medium.com/max/900/0*rA05n6siCddLinjn.png).

The loss falls rapidly for the first ten epochs, but thereafter tends to remain constant for a long time, as seen in Figure (a). Following that, the loss tends to reduce substantially, as illustrated in figure (b), before becoming practically constant.

Many of us may base our decision on the curve depicted in fig (a), however the fact is that if we train our network for additional epochs, there is a probability that the model will converge at a better position.

These plateaus complicate our judgement on when to stop the gradient drop and also slow down convergence because traversing a plateau in the expectation of minimising the loss demands more iterations.

## Cause of Plateau
The formation of a plateau is caused primarily by two factors, which are as follows:
* Saddle Point
* Local Minima

![img](https://cdn-images-1.medium.com/max/900/1*-ya2AEsB91XDsjXkMjs-tg.png)
[Image Credits](https://medium.com/r/?url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FDefinition-of-grey-level-blobs-from-local-minima-and-saddle-points-2D-case_fig1_10651758).

## Saddle Point

![img](https://cdn-images-1.medium.com/max/900/0*OQE_bSxccQ6R45P5.png)
[Image Credits](https://medium.com/r/?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSaddle_point)

The fundamental problem with saddle points is that the gradient of a function is zero at the saddle point, which does not reflect the greatest and minimum value. The gradient value optimises the machine learning and optimization algorithms in a neural network, and if the gradient is zero, the model becomes stalled.

## Local Minima

![img](https://cdn-images-1.medium.com/max/900/0*UfvC_Z1JespJIOcr.png)
[Image Credits](https://www.researchgate.net/figure/1st-order-saddle-point-in-the-3-dimensional-surface-Surface-is-described-by-the_fig7_280804948)

In this scenario, the point is an extremum, which is good, but the gradient is zero. We may not be able to escape the local minimum if our learning rate is too low. The loss value in our hypothetical training environment began balancing around some constant number, as shown in fig(a); one major explanation for this is the establishment of these types of local minimums.

## Effect of Learning Rate
The learning rate hyperparameter determines how quickly the model learns. A higher learning rate allows the model to learn faster, but it may result in a less-than-ideal final set of weights. A slower learning rate, on the other hand, may allow the model to acquire a more optimal, or possibly a globally ideal, set of weights, but training will be much more time consuming. A sluggish learning rate has the disadvantage of never convergent or becoming stuck on a suboptimal solution.

Thus, learning rate is important in overcoming the plateau problem; strategies such as scheduling the learning rate or cyclical learning rate are employed for this.

## Methods to Overcome a Plateau Problem
Following are the approaches which might be used to tweak the learning rates in order to overcome the plateau problem:

## Scheduling the Learning Rate
The most frequent method is to plan the learning rate, which suggests beginning with a reasonably high learning rate and gradually decreasing it over training. The concept is that we want to get from the initial parameters to a range of excellent parameter values as rapidly as possible, but we also want a low enough learning rate to explore the deeper, but narrower, regions of the loss function.

![img](https://cdn-images-1.medium.com/max/900/0*lv38Hvzb6PwX0ZNt.png)
[Image Credits](https://medium.com/r/?url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FStep-Decay-Learning-Rate_fig3_337159046)

An example of this is Step decay in which the learning rate is lowered by a certain percentage after a certain number of training epochs.

## Cyclical Learning Rate
Leslie Smith provided a cyclical learning rate scheduling approach with two bound values that fluctuate.

Cyclical learning scheme displays a sublime balance between passing over local minima while still allowing us to look around in detail.

![img](https://cdn-images-1.medium.com/max/900/0*lgTFEwR5GT2u2EX4.png)

[Image Credits](https://medium.com/r/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F1506.01186.pdf)

Thus, scheduling the learning rate aids us in overcoming the plateau issues encountered while optimising neural networks
## References
* [Analytics Vidhya](https://medium.com/r/?url=https%3A%2F%2Fanalyticsindiamag.com%2Fwhat-is-the-plateau-problem-in-neural-networks-and-how-to-fix-it%2F)
* [Plateau Phenomenon by Mark Ainsworth](https://medium.com/r/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F2007.07213.pdf)
* [Cyclical Learning Rate](https://medium.com/r/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F1506.01186.pdf)
* [Find best learning rate on Plateau](https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2FJonnoFTW%2Fkeras_find_lr_on_plateau)
* [Original Article on Plateau](https://www.vevesta.com/blog/13-Early-stopping-of-neural-network-might-not-be-optimal-decision-Plateau-problem?utm_source=GitHub_VevestaX_plateauProblem)

## Credits
[Vevesta](https://www.vevesta.com?utm_source=Github_VevestaX_Plateau) is Your Machine Learning Team's Collective Wiki: Save and Share your features and techniques. Explore [Vevesta](https://www.vevesta.com?utm_source=Github_VevestaX_Plateau) for free. For more such stories, follow us on twitter at [@vevesta1](http://twitter.com/vevesta1).

## Author
Sarthak Kedia

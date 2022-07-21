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

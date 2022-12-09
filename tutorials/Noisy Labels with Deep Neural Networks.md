
# Deep Dive into approaches for handling Noisy Labels with Deep Neural Networks

### Damaging effects of Noisy Labels on training and how best to manage them

## Why Noisy Labels are harmful to Neural Network ?

In machine learning tasks, such as computer vision, information retrieval, language processing, etc, more and better data means better results. Unreliable labels are called noisy labels because they may be corrupted from ground-truth labels. According to [authors](https://arxiv.org/pdf/2007.08199.pdf), the ratio of corrupted labels in real-world datasets is reported to range from 8.0% to 38.5%. Deep Neural Network learn from noisy labels as well as correctly labelled data and this results in poor generalizability of the models. Deep learning is more susceptible to label noises than traditional machine learning owing to its high expressive power. Also, achieving a good generalization capability in the presence of noisy labels becomes a challenge since the accuracy drop with label noise is considered to be more harmful than with other noises, such as input noise.

## What are the Types of Noise ?

According to the [authors](https://arxiv.org/pdf/2007.08199.pdf), noise present in the labels of the supervised data is of following types:

1. Instance-independent Label Noise: A typical approach for modeling label noise assumes that the label corruption process is conditionally independent of data features when the true label is given
2. Instance-dependent Label Noise: In more realistic settings, the label corruption probability is assumed to be dependent on both the data features and class labels.

## What Conventional Approaches can be used to manage Noisy Labels ?

According to [authors](https://arxiv.org/pdf/2007.08199.pdf), following are the non-deep learning approaches that can be used to manage noisy labels:

#### 1. Data Cleaning :
Training data is cleaned by excluding exclude false labeled examples from noisy training data. Some techniques that are used are bagging, boosting, k-means neighbour, outlier detection and anomaly detection.

#### 2. Use of Convex Surrogate Function :
According to [authors](https://arxiv.org/pdf/2007.08199.pdf), “Convex surrogate loss functions, which approximate the 0-1 loss function, have been proposed to train a specified classifier under the binary classification setting. However, these loss functions cannot support the multi-class classification task.”

#### 3. Probabilistic Method :
In family of methods, the confidence of each label is estimated by clustering and then used for a weighted training scheme. This confidence is converts hard labels into soft labels which reflects the uncertainty of labels. However, this family of methods may exacerbate the overfitting issue owing to the increased number of model parameters.

#### 4. Model-based Method:
Model modfications have been proposed for techniques like SVM and decision tree to make them robust to noisy data.

## What Deep Learning Approaches can be used to manage Noisy Labels ?

According to [authors](https://arxiv.org/pdf/2007.08199.pdf), following techniques can be used:

#### 1. Noise Adaptation Layer:
From the view of training data, the noise process is modeled by discovering the underlying label transition pattern. Example, Webly learning [2] first trains the base DNN only for easy examples and subsequently, the confusion matrix for all training examples is used as the initial weight W of the noise adaptation layer. According to authors [1], “a common drawback of this family is their inability to identify false-labeled examples, treating all the examples equally. Thus, the estimation error for the transition matrix is generally large when only noisy training data is used or when the noise rate is high”.

#### 2. Probabilistic Noise Modeling :
According to authors[1], this model manages two independent networks, each of which is specialized to predict the noise type and label transition probability. Both networks are trained with massive noisy labeled data after the pretraining step with a small amount of clean data.

#### 3. Contrastive-Additive Noise Network
This network introduced a new concept of quality embedding, which models the trustworthiness of noisy labels.

#### 4. Regularization Techniques
Regularization techniques such as data augmentation, weight decay, dropout, and batch normalization. These regularization methods operate well on moderately noisy data, but they alone do not sufficiently improve the test accuracy. Also, poor generalization could be obtained when the noise is heavy.

#### 5. Pre-training
According to authors [1, 3], empirically proves that fine-tuning on a pre-trained model provides a significant improvement in robustness compared with models trained from scratch. The universal representations of pre-training prevent the model parameters from being updated in the wrong direction by noisy labels.

#### 6. PHuber
According to authors [1,4], PHuber is a composite loss-based gradient clipping for label noise robustness.

#### 7. Adversarial training
This technique enhances the noise tolerance by encouraging the DNN to correctly classify both original inputs and hostilely perturbed ones.

#### 8. Label smoothing
This technique estimates the marginalized effect of label noise during training, thereby reducing overfitting by preventing the DNN from assigning a full probability to noisy training examples. Instead of the one-hot label, the noisy label is mixed with a uniform mixture over all possible labels.

#### 9. Noise-Robust Loss Functions
Inorder to define noise-robust loss functions modifications were made to known loss functions. Noise-robust loss functions are generalized cross entropy [18], symmetric cross entropy [19], curriculum loss [20], active passive loss [21], etc. According to authors[1,5], these loss functions perform well only in simple cases, when learning is easy or the number of classes is small. Moreover, the modification of the loss function increases the training time for convergence.

#### 10. Loss Adjustment
According to authors[1], by changing the loss of all training instances before updating the DNN, loss adjustment is useful for minimising the detrimental effects of noisy labels. Techniques such as Backward correction [14], Forward correction [14], Gold standard correction [17], Dynamic Bootstrapping [15], Self-adaptive training [16] falls under this category of solutions.

## Some Github repositories of Noise-Robust techniques

List of some solutions on Github meant to handle noisy data are below:

1. Noise Model: Training convolution networks with Noisy labels by authors[5]. The Keras code is present in Github link [7] .
2. Pre-Training: Using pre-training can improve model robustness and uncertainty by authors[3]. The Pytorch code is present in Github link [8].
3. Probabilistic Noise Model: Learning from massive noisy labeled data for image classification by authors[9]. The Caffe implementation is present in Github [10].
4. PHuber: “Can gradient clipping mitigate label noise?” by authors [4]. The Pytorch implementation is present Github [11].
5. Adversarial Training: By authors[12] with Pytorch implementation present in Github [13].

## References:

1. [Survey on Learning from Noisy Labels](https://arxiv.org/pdf/2007.08199.pdf)
2. A. J. Bekker and J. Goldberger, “Training deep neural-networks based on unreliable labels,” in Proc. ICASSP, 2016, pp. 2682–2686.
3. D. Hendrycks, K. Lee, and M. Mazeika, “Using pre-training can improve model robustness and uncertainty,” in Proc. ICML, 2019.
4. A. K. Menon, A. S. Rawat, S. J. Reddi, and S. Kumar, “Can gradient clipping mitigate label noise?” in Proc. ICLR, 2020.
5. S. Sukhbaatar, J. Bruna, M. Paluri, L. Bourdev, and R. Fergus, “Training convolutional networks with noisy labels,” in Proc. ICLRW, 2015.
6. M. Ren, W. Zeng, B. Yang, and R. Urtasun, “Learning to reweight examples for robust deep learning,” in Proc. ICML, 2018.
7. [Github Link for Keras implemetation of Noise Model](https://github.com/delchiaro/training-cnn-noisy-labels-keras)
8. [Github Link for Pytorch implementation of pre-training](http://1github.com/hendrycks/pre-training)
9. T. Xiao, T. Xia, Y. Yang, C. Huang, and X. Wang, “Learning from massive noisy labeled data for image classification,” in Proc. CVPR, 2015, pp. 2691–2699.
10. [Github link for Caffe implementation of Probabilistic Noise Model](https://github.com/Cysu/noisy_label)
11. [Github link for Pytorch implementation of PHuber](http://2https//github.com/dmizr/phuber)
12. I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in Proc. ICLR, 2014.
13. [Github Link for Pytorch implementation of Adversial training](http://5https//https://github.com/sarathknv/adversarial-examples-pytorch)
14. G. Patrini, A. Rozza, A. Krishna Menon, R. Nock, and L. Qu, “Making deep neural networks robust to label noise: A loss correction
15. D. Hendrycks, M. Mazeika, D. Wilson, and K. Gimpel, “Using trusted data to train deep networks on labels corrupted by severe noise,” in Proc. NeurIPS, 2018, pp. 10 456–10 465.
16. E. Arazo, D. Ortego, P. Albert, N. E. O’Connor, and K. McGuinness, “Unsupervised label noise modeling and loss correction,” in Proc. ICML, 2019
17. L. Huang, C. Zhang, and H. Zhang, “Self-adaptive training: beyond empirical risk minimization,” in Proc. NeurIPS, 2020.
18. Z. Zhang and M. Sabuncu, “Generalized cross entropy loss for training deep neural networks with noisy labels,” in Proc. NeurIPS, 2018, pp. 8778–8788.
19. Y. Wang, X. Ma, Z. Chen, Y. Luo, J. Yi, and J. Bailey, “Symmetric cross entropy for robust learning with noisy labels,” in Proc. ICCV, 2019, pp. 322–330.
20. Y. Lyu and I. W. Tsang, “Curriculum loss: Robust learning and generalization against label corruption,” in Proc. ICLR, 2020.
21. X. Ma, H. Huang, Y. Wang, S. Romano, S. Erfani, and J. Bailey, “Normalized loss functions for deep learning with noisy labels,” in Proc. ICML, 2020, pp. 6543–6553.
22. [Noisy Labels with Deep Neural Networks article on Vevesta](https://www.vevesta.com/blog/24-Handling-Noisy-Labels-Neural-Network)
23. [Noisy Labels with Deep Neural Networks article on Substack](https://vevesta.substack.com/p/deep-dive-into-approaches-for-handling)
## Credits

The above article is sponsored by [vevesta](https://www.vevesta.com/).

[Vevesta](https://www.vevesta.com/): Your Machine Learning Team’s Feature and Technique Dictionary: Accelerate your Machine learning project by using features, techniques and projects used by your peers. Explore [Vevesta](https://www.vevesta.com/) for free. For more such stories, follow us on twitter at [@vevesta_labs](https://twitter.com/vevesta_labs).

# Why is Everyone Training Very Deep Neural Network with Skip Connections?

Deep neural networks (DNNs) have are a powerful means to train models on various learning tasks, with the capability to automatically learn relevant features. According to empirical studies, there seem to be positive correlation between model depth and generalization performance.

Generally, training PlainNets (Neural networks without Skip Connections) with few number of layers (i.e. typically one to ten layers) is not problematic. But when model depth is increased beyond 10 layers, training difficulty can experienced. Training difficulty typically worsens with increase in depth, and sometimes even the training set cannot be fitted. For example, when training from scratch there was optimization failure for the VGG-13 model with 13 layers and VGG-16 model with 16 layers. Hence, VGG-13 model was trained by initializing its first 11 layers with the weights of the already trained VGG-11 model. Similar was the case with VGG-16. Currently, there is proliferation of networks, such as Resnet, FractalNet, etc which use skip connections.

## What are skip connections?

Skip connections are where the outputs of preceding layers are connected (e.g. via summation or concatenation) to later layers. Architectures with more than 15 layers have increasingly turned to skip connections. According to empirical studies, skip connections alleviate training problems and improve model generalization. Although multiple weights initialization schemes and batch normalization can alleviate the training problems, optimizing PlainNets becomes absolutely impossible beyond a certain depth.

## Experimental Results

Experiments were done on MNIST, CIFAR-10 and CIFAR-100 datasets using PlainNet, ResNet and ResNeXt, each having 164 layers.

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9bacd8c9-0a4f-4bc0-ba1a-ca039a5e8186_502x582.png)

Tables 1, 2, 3 and 4 show the obtained accuracies on the different datasets. Clearly it can be seen, as in figure 3 and figure 4, that PlainNets perform worser than networks with skip connections and are essentially untrainable. PlainNets failure to learn, given the very poor accuracies on the training sets.

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F9bacd8c9-0a4f-4bc0-ba1a-ca039a5e8186_502x582.png)

## Discussion and Observations:

The plot of PlainNets activations and weights given below in Figure 5 and Figure 6.

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F65451c79-d401-4c78-bbae-977c3b332df7_1040x391.png)


![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F3c4e32a0-3194-4ca9-9687-32df9f90ed4b_993x387.png)
The plot of ResNet unit’s activations and weights given below in Figure 7 and Figure 8.

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc1259c96-7dd4-4225-855a-4aba70865ecd_977x375.png)

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffd6e1a1d-5e41-44d0-882c-951ed742caac_1014x418.png)

According to authors of the [paper](https://orbilu.uni.lu/bitstream/10993/48927/1/TNNLS-2020-P-13752.pdf), the PlainNet trained on CIFAR10 dataset, starting from the eightieth layer, have hidden representations with infinite condition numbers; on CIFAR100 dataset, starting from the hundredth layer, the PlainNet’s hidden representations have infinite condition numbers. This observation depicts the worst scenario of the singularity problem for optimization such that model generalization is impossible as given in Remark 8. In contrast, the hidden representations of the ResNet never have infinite condition numbers; the condition numbers, which are high in the early layers quickly reduce to reasonable values so that optimization converges successfully.

## Conclusion

Skip connections are a powerful means to train Deep Neural Networks.

## Credits

The above article is sponsored by [Vevesta](https://www.vevesta.com/).

[Vevesta](https://www.vevesta.com/): Your Machine Learning Team’s Feature and Technique Dictionary: Accelerate your Machine learning project by using features, techniques and projects used by your peers. Explore [Vevesta](https://www.vevesta.com/) for free. For more such stories, follow us on twitter at [@vevesta_labs](https://twitter.com/vevesta_labs).

100 early birds who login into [Vevesta](https://www.vevesta.com/) will get free subscription for 3 months

Subscribe to receive a copy of our newsletter directly delivered to your inbox.
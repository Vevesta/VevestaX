
# Why should you be looking at Few-Shot Learning?

Unlike Transfer Learning, Few Shot learning is more about training unfamiliar tasks and optimizing the hyperparameters for networks that are not trained at all

In a fast-growing society, speed and efficiency are widely appreciated by people especially when it's accompanied by sure-found results. At the same age where human speed and flexibility are praised coherently machines with the ability to adapt and grow to the constant change in data are considered to be crucial for such a rapid moving time. In the machine learning world, Reptile is one of such quick adapting training algorithms that can be applied to learning problems and models by meta-learning which can be imagined as a machine learning to learn better for faster and more accurate computation.


## What is few-shot learning?

As the name suggests few-shot learning is a sub-genre of machine learning in which a model is trained in an objective aspect so that it can swiftly get used to any given task with limited data within a small number of training repetitions. This method to train a model is generally used to classify or distinguish objects from the given data samples. Few-shot learning can also be called as One shot or Zero Shot learning depending on the number of data samples used to train the model.

To gain deeper insight into Reptile it is essential that we understand what is Model Agnostic Meta-Learning.

## Model Agnostic Meta-Learning

Model Agnostic Meta-Learning or MAML, a few-shot algorithm, that can learn a new task with the small amount of data presented to it, and fine-tunes the parameters of any standard model via meta-learning to enhance that model for smoother adaptation. Being called model agnostic, it can be directly applied to any learning problem and model that is trained with a gradient descent procedure. The gradient descent procedure in machine learning and is used to reduce the cost/loss function. It’s easily applicable and hence considered one of the fundamentals to learn in machine learning.

## The Working of MAML Algorithms

To comprehend the working of MAML algorithms we must use the figure below to visualize the approach this algorithm uses.

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5ed2b75b-3bf7-4691-8145-f931b31342ba_341x190.png)

We have an initial set of parameters θ and ▽L1,▽L2, and ▽L3 are new tasks. We can visualize each meta-iteration as a branch of a tree and each task attached can be seen as the twigs from the branches and the data provided for the task being the leaves, the end goal is updating the tree’s parameters to feint towards the sun for maximum production of energy.

So to obtain better results on a new task we must learn the prime set of initial parameters by each meta-iteration and train the model’s initial parameters such that we get closer to our result from the least amount of gradient steps adapting rapidly to each of the new classification tasks that will then be fine-tuned as per the favorable inner parameter by meta-learning, by each inner iteration of the task and would result in a crystal clear classification of the tasks given.

This from a feature learning standpoint is building an internal representation or input to output mapping that our system can hold and make so that it’s susceptible to managing many tasks altogether. If they are favorable then just fine-tuning the parameters gives us better results. In other words, the algorithm maximizes the sensitivity of the loss functions from the tasks, changing and optimizing the parameter to bring improvements in the task loss.

## How MAML Differs From Transfer Learning

* Transfer learning aims at improving the process of learning new tasks using the experience and knowledge gained by solving a predecessor problem that is somewhat similar. It takes a more creatural approach by absorbing and applying to solve problems, constantly training from similar tasks.

* While MAML being a unit of its own applies its algorithms to any given data presented and adapts to the number of information fed to it. Much like a shapeshifter, it basically shifts and learns to train itself towards the outcome through each repetition. It’s more about speeding up and optimizing the hyperparameters for networks that are not trained at all.

* MAML has a slight upper hand as it constantly evolves to the task presented all while maintaining its speed and fluidity, while Transfer Learning bogs down when presented an unfamiliar task which makes MAML more dependable.

## Introduction to Reptile

The Reptile algorithm was developed by OpenAI to perform model agnostic meta-learning. Reptile is a simplification of MAML, obtained by ignoring second-order derivatives. This algorithm was specially designed with the intention to execute a task rapidly by repeatedly sampling a task, training on it, and moving the initialization towards the trained weights on that task. Just like MAML, Reptile works by performing stochastic gradient descent using the difference between parameters trained on a batch of fresh data and the model parameters prior to training over a fixed number of meta-iterations. Which proves to give outstanding results for few-shot learning. We use the Algorithm on an omniglot dataset and check the results.

## Implementing of Reptile on Omniglot Dataset

We look at the execution of Reptile

```
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
```

## Defining The Hyperparameters

```
learning_rate = 0.003
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 2000
eval_iters = 5
inner_iters = 4

eval_interval = 1
train_shots = 20
shots = 5
classes = 5
```

## The Omniglot dataset

The Omniglot dataset is a dataset of 1,623 characters taken from 50 different alphabets, with 20 different handwritten examples for each character. For the few-shot learning task, k samples from the 20 examples for each character are chosen randomly from n randomly-chosen classes from the class list. These n values are used to create a new set of temporary labels to use to test the model's ability to learn a new task given only a few examples. In other words, if you are training on 5 classes, your new class labels will be one among 0, 1, 2, 3, or 4.

```
class Dataset:
# This class will facilitate the creation of a few-shot dataset
# from the Omniglot dataset that can be sampled from quickly while also
# allowing to create new labels at the same time.
def __init__(self, training):
# Download the tfrecord files containing the omniglot data and convert to a
# dataset.
split = "train" if training else "test"
ds = tfds.load("omniglot", split=split, as_supervised=True, shuffle_files=False)
# Iterate over the dataset to get each individual image and its class,
# and put that data into a dictionary.
self.data = {}

def extraction(image, label):
    # This function will shrink the Omniglot images to the desired size,
    # scale pixel values and convert the RGB image to grayscale
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [28, 28])
    return image, label

for image, label in ds.map(extraction):
    image = image.numpy()
    label = str(label.numpy())
    if label not in self.data:
        self.data[label] = []
    self.data[label].append(image)
self.labels = list(self.data.keys())

def get_mini_dataset(
self, batch_size, repetitions, shots, num_classes, split=False
):
temp_labels = np.zeros(shape=(num_classes * shots))
temp_images = np.zeros(shape=(num_classes * shots, 28, 28, 1))
if split:
    test_labels = np.zeros(shape=(num_classes))
    test_images = np.zeros(shape=(num_classes, 28, 28, 1))

# Get a random subset of labels from the entire label set.
label_subset = random.choices(self.labels, k=num_classes)
for class_idx, class_obj in enumerate(label_subset):
    # Use enumerated index value as a temporary label for mini-batch in
    # few shot learning.
    temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
    # If creating a split dataset for testing, select an extra sample from each
    # label to create the test dataset.
    if split:
        test_labels[class_idx] = class_idx
        images_to_split = random.choices(
            self.data[label_subset[class_idx]], k=shots + 1
        )
        test_images[class_idx] = images_to_split[-1]
        temp_images[
            class_idx * shots : (class_idx + 1) * shots
        ] = images_to_split[:-1]
    else:
        # For each index in the randomly selected label_subset, sample the
        # necessary number of images.
        temp_images[
            class_idx * shots : (class_idx + 1) * shots
        ] = random.choices(self.data[label_subset[class_idx]], k=shots)

dataset = tf.data.Dataset.from_tensor_slices(
    (temp_images.astype(np.float32), temp_labels.astype(np.int32))
)
dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
if split:
    return dataset, test_images, test_labels
return dataset

import urllib3

urllib3.disable_warnings()  # Disable SSL warnings that may happen during download.
train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)
```

## Visualizing the dataset

```
_, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

sample_keys = list(train_dataset.data.keys())

for a in range(5):
  for b in range(5):
      temp_image = train_dataset.data[sample_keys[a]][b]
      temp_image = np.stack((temp_image[:, :, 0],) * 3, axis=2)
      temp_image *= 255
      temp_image = np.clip(temp_image, 0, 255).astype("uint8")
      if b == 2:
          axarr[a, b].set_title("Class : " + sample_keys[a])
      axarr[a, b].imshow(temp_image, cmap="gray")
      axarr[a, b].xaxis.set_visible(False)
      axarr[a, b].yaxis.set_visible(False)
plt.show()
```

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https://bucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com/public/images/88e75150-0ee4-4b37-84aa-0736354270e3_1128x1121.png)

## Building the model

```
def conv_bn(x):
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

inputs = layers.Input(shape=(28, 28, 1))
x = conv_bn(inputs)
x = conv_bn(x)
x = conv_bn(x)
x = conv_bn(x)
x = layers.Flatten()(x)
outputs = layers.Dense(classes, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile()
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
```

## Training the model

```
training = []
testing = []
for meta_iter in range(meta_iters):
frac_done = meta_iter / meta_iters
cur_meta_step_size = (1 - frac_done) * meta_step_size
# Temporarily save the weights from the model.
old_vars = model.get_weights()
# Get a sample from the full dataset.
mini_dataset = train_dataset.get_mini_dataset(
inner_batch_size, inner_iters, train_shots, classes
)
for images, labels in mini_dataset:
with tf.GradientTape() as tape:
    preds = model(images)
    loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
grads = tape.gradient(loss, model.trainable_weights)
optimizer.apply_gradients(zip(grads, model.trainable_weights))
new_vars = model.get_weights()
# Perform SGD for the meta step.
for var in range(len(new_vars)):
new_vars[var] = old_vars[var] + (
    (new_vars[var] - old_vars[var]) * cur_meta_step_size
)
# After the meta-learning step, reload the newly-trained weights into the model.
model.set_weights(new_vars)
# Evaluation loop
if meta_iter % eval_interval == 0:
accuracies = []
for dataset in (train_dataset, test_dataset):
    # Sample a mini dataset from the full dataset.
    train_set, test_images, test_labels = dataset.get_mini_dataset(
        eval_batch_size, eval_iters, shots, classes, split=True
    )
    old_vars = model.get_weights()
    # Train on the samples and get the resulting accuracies.
    for images, labels in train_set:
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    test_preds = model.predict(test_images)
    test_preds = tf.argmax(test_preds).numpy()
    num_correct = (test_preds == test_labels).sum()
    # Reset the weights after getting the evaluation accuracies.
    model.set_weights(old_vars)
    accuracies.append(num_correct / classes)
training.append(accuracies[0])
testing.append(accuracies[1])
if meta_iter % 100 == 0:
    print(
        "batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
    )
```

## Output

```
batch 0: train=0.000000 test=0.600000
batch 100: train=0.600000 test=0.800000
batch 200: train=1.000000 test=0.600000
batch 300: train=0.600000 test=0.800000
batch 400: train=0.800000 test=1.000000
batch 500: train=1.000000 test=0.600000
batch 600: train=1.000000 test=1.000000
batch 700: train=1.000000 test=1.000000
batch 800: train=1.000000 test=0.600000
batch 900: train=1.000000 test=1.000000
batch 1000: train=0.800000 test=1.000000
batch 1100: train=1.000000 test=0.600000
batch 1200: train=0.800000 test=1.000000
batch 1300: train=0.800000 test=1.000000
batch 1400: train=1.000000 test=1.000000
batch 1500: train=0.800000 test=1.000000
batch 1600: train=1.000000 test=1.000000
batch 1700: train=1.000000 test=0.800000
batch 1800: train=1.000000 test=1.000000
batch 1900: train=0.800000 test=1.000000
```

As we can see here our model is gradually getting better at the new test set and adapting to a new classification problem. Thus we see that the few-shot learning through Reptile gives you definite results in fewer computations.

## Conclusion

* The Reptile Algorithm learns the optimal set of initial parameters for adapting to new classification tasks in a minimum number of shots or sampled data going through less number of training rounds making it the most pre-eminent choice of few-shot Learning technique.
* It is more involving for a Machine Learning apprentice and a detrimental step to understanding other complex learning methods which apply the teacher-student inner-outer loop.
* Has more space for research since it's one of the few fast-responding learning techniques with fewer number of samples given.
* Building more generalized models which can learn to solve whole arrays of tasks, making it more versatile and dependable.

## References:

1. [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)
2. [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)
3. [Few-shot classification of the Omniglot dataset using Reptile.](https://keras.io/examples/vision/reptile/)
4. [Towards Data Science](https://towardsdatascience.com/icml-2018-advances-in-transfer-multitask-and-semi-supervised-learning-2a15ef7208ec)
5. [transfer-learning-machine-learning](http://transfer-learning-machine-learning/)
6. [Few-Shot Learning with Reptile - Keras Code Examples](https://www.youtube.com/watch?v=qEHZ1DeBF-M&ab_channel=ConnorShorten)
7. [Few-Shot Learning Article on Vevesta](https://www.vevesta.com/blog/22-Few-Shot-Learning)
8. [Few-Shot Learning Article on Substack](https://vevesta.substack.com/p/why-should-you-be-looking-at-few)

## Credits:

The above article is sponsored by [vevesta.](https://www.vevesta.com/)

[Vevesta](https://www.vevesta.com/): Your Machine Learning Team’s Feature and Technique Dictionary: Accelerate your Machine learning project by using features, techniques and projects used by your peers. Explore [Vevesta](https://www.vevesta.com/) for free. For more such stories, follow us on twitter at [@vevesta_labs](https://twitter.com/vevesta_labs).
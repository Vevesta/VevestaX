
# Deep Dive into Attention Network

If we are providing a huge dataset to the model to learn, it is possible that a few important parts of the data might be ignored by the models. Paying attention to important information is necessary and it can improve the performance of the model. This can be achieved by adding an additional attention feature to the models.

Similar to in real-life, the Attention in neural networks also refers to the most important details one should focus on (or attend to) in order to solve a given task. The goal of introducing Attention in Deep learning is to teach the machine where to pay attention to, given its purpose and context.

We can introduce an attention mechanism to create a shortcut between the entire input and the context vector where the weights of the shortcut connection can be changeable for every output. Because of the connection between input and context vector, the context vector can have access to the entire input, and the problem of forgetting long sequences can be resolved to an extent.

In the influential paper [Show, Attend and Tell, Kelvin Xu et. al.](https://arxiv.org/pdf/1502.03044v2.pdf) introduce Attention to a Recurrent neural network to generate captions for images. The words of the caption are generated one-by-one, for each word, the model pays attention to a different part of the image.

![](https://cdn-images-1.medium.com/max/1000/0*PBTHyHSGDI7Qk3ZL.png)

The figure above illustrates their result. The underlined words are the words that the model generates at that step, the brighter regions show where the model attends to generate those words.

For the sake of understanding Attention, let's generate a synthesis dataset and train a network to estimate its Attention function using PyTorch.

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```

First, let's generate a synthesis dataset, whose input data is of sequential type. There will be 10.000 data points, each is a sequence of 8 floats.

```
input_size = 10000
seq_len = 8
inputs = torch.rand((input_size, seq_len))
display(inputs)
print(f'input shape: {inputs.shape}')
```

Next, we need to define the contexts. Here, to make things clearer, we separate contexts from the inputs, that is to say, we define the contexts independently. There are 5 different contexts, indexed from 0 to 4. For each input, there is one corresponding context.

```
n_contexts = 5
context = torch.randint(
    low=0, high=n_contexts, size=(input_size, 1))
display(context)
print(f'context shape: {context.shape}')  
```

Now, we need to establish a connection between the contexts and the outputs. If there is no dependency between the contexts and the outputs, the whole point of attention is lost. Return to this dataset, we make it so that the output given an input sequence is equal to a value in that sequence, the corresponding context determines which value (in the sequence of 8 values) that is.

```
true_attention = {
    0:2,
    1:7,
    2:3,
    3:5,
    4:1
}
true_attention
```

While the true_attention is a dictionary, mapping from a context value to the position in the input sequence that the output should mimic. Note that this is the ground truth that our Attention network does not know about and is trying to approximate. This means if the context equals 0, then the model should pay all attention to the 2nd value of the input, if the context is 1, then all attention should be on the 7th value of the input, and so on. We generate the outputs accordingly.

```
outputs = torch.tensor([
    inputs[i, true_attention[context[i].item()]]
    for i in range(input_size)
])
display(outputs)
print(f'output shape: {outputs.shape}')
```

The dataset is ready, we then build the network. The Attention network is very simple. It has an Embedding layer for the context (this is where the network will learn how contexts affect Attention) and a Linear layer that computes the output from the attention glimpse. For training, each time a pair of (input, context) is fed to the network, it embeds the context to get the Attention, multiplies the input with the Attention to get the attention glimpse, and then passes the attention glimpse through the Linear layer to produce the prediction. The loss is then computed and backpropagates through the network to update the weights, as usual.

```
class AttentionNetwork(nn.Module):
    def __init__(self):
        super(AttentionNetwork, self).__init__()
        self.context_embed = nn.Embedding(n_contexts, seq_len)
        self.linear = nn.Linear(seq_len, 1)

    def forward(self, x, c): # x is input (feature), c is context
        a = self.context_embed(c)
        x = x * a # element-wise multiplication
        x = self.linear(x)
        return x

    def get_attention(self, c):
        a = self.context_embed(c)
        return a
model = AttentionNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
```

The function get_attention is there to provide us the network's computed Attention for a given context. We will call this function later, when all training is done.

```
model.train()
for epoch in range(4):
    losses = []
    for i in tqdm(range(input_size)):
        inp = inputs[i]
        c = context[i]
        optimizer.zero_grad()
        pred = model(inp, c).squeeze()
        loss = criterion(pred, outputs[i])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f'epoch {epoch}: MSE = {np.mean(losses):.7f}')
```

After 4 epochs, the model seems to has converged. The mean squared error is quite small with four 0s after the floating-point. Let us see if the network has approximated the ground truth attention right. For this purpose, we draw a plot that consists of 5 subplots, each represents a context. In a subplot, there is a green bar with height 1 showing the ground truth attention of that context, while the normalized attention approximation of the network is shown using orange bars.

```
model.eval()
fig, ax = plt.subplots(n_contexts, figsize=(15, 10))
for c in range(n_contexts):
    true_att_index = np.zeros(seq_len)
    true_att_index[true_attention[c]] = 1
    ax[c].bar(range(seq_len),true_att_index, color='green')

    computed_attention = model.get_attention(torch.tensor(c)).detach().abs()
    computed_attention /= computed_attention.sum()
    ax[c].bar(range(seq_len), computed_attention, color='orange')
```

![](https://cdn-images-1.medium.com/max/1000/0*mAlMphHCJK4GqEX7.png)

We can see that the network has learned pretty well, most of the green bars are filled with orange. Actually, if we let the training continue for several more epochs, there would be hardly any green on the plot, since the network would have approximated the attention function almost perfectly.

## References:
1. [Attention in Deep Learning](https://github.com/Mothaiba/Attention-in-Deep-Learning-your-starting-point/blob/main/Attention-synthesis-example.ipynb)
2. [Attention Network article on Vevesta.com](https://www.vevesta.com/blog/19-Attention-Network)
3. [Attention Network article on Substack](https://vevesta.substack.com/p/attention-network-deeper-look-into)

## Credits

The above article is sponsored by [Vevesta](https://www.vevesta.com/).

[Vevesta](https://www.vevesta.com/): Your Machine Learning Team’s Feature and Technique Dictionary: Accelerate your Machine learning project by using features, techniques and projects used by your peers. Explore [Vevesta](https://www.vevesta.com/) for free. For more such stories, follow us on twitter at [@vevesta_labs](https://twitter.com/vevesta_labs).

100 early birds who login into [Vevesta](https://www.vevesta.com/) will get free subscription for 3 months

Subscribe to receive a copy of our newsletter directly delivered to your inbox.
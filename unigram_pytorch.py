"""Pytorch."""
import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
from matplotlib import pyplot as plt
from collections import defaultdict

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def logit(x: FloatArray) -> FloatArray:
    """Compute logit (inverse sigmoid)."""
    return np.log(x) - np.log(1 - x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vector so that it sums to 1."""
    return x / torch.sum(x)


def loss_fn(p: float) -> float:
    """Compute loss to maximize probability."""
    return -p


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct initial s - corresponds to uniform p
        s0 = logit(np.ones((V, 1)) / V)
        self.s = nn.Parameter(torch.tensor(s0.astype("float32")))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        p = normalize(torch.sigmoid(self.s))

        # compute log probability of input
        # print(f"input is: {input}")
        # print(f"torch sum of input is:{torch.sum(input,1,keepdim=True)}")
        # exit()
        # print(torch.sum(input, 1, keepdim=True))
        # print((torch.sum(input, 1, keepdim=True)).dtype)
        return torch.sum(input, 1, keepdim=True).T @ torch.log(p), p


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]
    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]
    token_array = np.array(tokens)
    complete_vocab_np = np.unique(token_array)
    complete_vocab = list(complete_vocab_np)
    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])
    encodings1 = np.hstack([onehot(complete_vocab, token) for token in tokens])
    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype("float32"))
    # print(x.shape)
    # print(f"input: {x}")
    # x1 = torch.tensor(encodings1.astype("float32"))
    # define model
    model = Unigram(len(vocabulary))
    # model1 = Unigram(len(complete_vocab))
    # set number of iterations and learning rate
    num_iterations = 500
    learning_rate = 0.1

    loss_vals = []
    loss1_vals = []
    runs = []
    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    for i in range(num_iterations):
        runs.append(i)
        # p_pred1, p1 = model1(x1)
        p_pred, p = model(x)
        # loss1 = -p_pred1
        loss = -p_pred
        # print(f"loss: {loss}")
        # loss1.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        # loss1_vals.append(np.log(loss1.item()))
        loss_vals.append(loss.item())
        # optimizer1.step()
        # optimizer1.zero_grad()
        optimizer.step()
        optimizer.zero_grad()

    # compute true probabilities
    char_dict = defaultdict(str)
    for char in tokens:
        char_dict[char] = char_dict.get(char, 0) + 1
        pass

    # Compute true prob for each vocab word
    true_prob = {
        char: (count / len(tokens))
        for char, count in char_dict.items()
        if char in vocabulary
    }
    # compute the prob of OOV words (in total)
    true_prob[None] = sum(
        count for char, count in char_dict.items() if char not in vocabulary
    ) / len(tokens)

    true_array = np.array([true_prob[char] for char in vocabulary])
    model_prob = p.detach().numpy().flatten()
    delta = np.absolute(true_array - model_prob)
    delta_sum = np.sum(delta)

    # Use previous code to compute minimum possible loss
    true_tensor = torch.reshape(torch.from_numpy(true_array), (28, 1))
    true_tensor_32 = true_tensor.to(torch.float32)

    one_hot_sum = torch.reshape(torch.sum(x, 1, keepdim=True).T, (1, 28))
    min_loss = one_hot_sum @ torch.log(true_tensor_32)
    run_array = np.array(runs)
    print(min_loss)
    min_loss_array = np.full(len(run_array), -min_loss)
    # display results
    loss_array = np.array(loss_vals)

    loss1_array = np.array(loss1_vals)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(run_array, loss_array)
    ax.plot(run_array, min_loss_array)
    label = "Minimum Loss = 1.9565e+06"
    # place the label in the lower right corner of the plot
    ax.annotate(
        label,  # this is the text
        (130, 1.9565e06),  # this is the point to label
        textcoords="offset points",  # how to position the text
        xytext=(0, 10),  # distance from text to points (x,y)
        ha="center",
    )  # horizontal alignment can be left, right or center
    ax.set_xlabel("Runs")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Loss vs Number of Gradient Descent Runsr (lr = {learning_rate},runs = {num_iterations-1})"
    )
    ax.legend()

    bar_ax = ax.inset_axes([0.4, 0.55, 0.6, 0.4])
    bar_width = 0.5
    plot_vocab = vocabulary
    plot_vocab[-1] = "N"
    plot_vocab[-2] = "_"
    x_pos = np.arange(len(vocabulary))
    bar1 = bar_ax.bar(x_pos, model_prob, bar_width, label="Predicted Probabilities")
    bar2 = bar_ax.bar(
        x_pos + bar_width, true_array, bar_width, label="True Probabilities"
    )
    bar_ax.set_xlabel("Characters in Vocabulary")
    bar_ax.set_ylabel("Probabilities")
    bar_ax.set_title("True vs Predicted Unigram Predictions")
    bar_ax.set_xticks(x_pos + 0.5 * bar_width, plot_vocab)
    bar_ax.text(
        10,
        0.15,
        f"Total Probability Error: {delta_sum:.3e}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
    )
    plt.show()


if __name__ == "__main__":
    gradient_descent_example()

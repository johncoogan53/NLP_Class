"""Demonstrate the "simplest" language model."""
import math
from typing import Optional

# generate English document
text = "Four score and seven years ago, our fathers brought forth, upon this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal."
text = (text + " ") * 100  # make the document very long

# tokenize - split the document into a list of little strings
tokens = [char for char in text]

# encode as a one-hot vector, check out ord() and chr() for more info on this comprehension
vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]


def onehot(vocabulary, char):
    """Check out this embedding comprehension too: for 0 in _"""
    embedding = [0 for _ in range(len(vocabulary))]
    try:
        idx = vocabulary.index(char)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx] = 1
    return embedding


encodings = [
    onehot(vocabulary, token) for token in tokens
]  # embed each character in the vocabulary as a vector with a 1 location based on its order in the vocabulary


# define model
class SimplestModel:
    """The simplest language model."""

    def __init__(self) -> None:
        """Initialize."""
        self.p_space: Optional[float] = None

    def train(self, encodings: list[int]) -> "SimplestModel":
        """Train the model on data."""
        self.p_space = encodings.count(0) / len(encodings)
        return self

    def apply(self, encodings: list[int]) -> float:
        """Compute the probability of a document."""
        if self.p_space is None:
            raise ValueError("This model is untrained")
        return sum(
            math.log(self.p_space) if encoding == 0 else math.log(1 - self.p_space)
            for encoding in encodings
        )


# train model
model = SimplestModel()
model.train(encodings)

# compute probability
log_p = model.apply(encodings)

# print
print(f"learned p_space value: {model.p_space}")
print(f"log probability of document: {log_p}")
print(f"probability of document: {math.exp(log_p)} (this is due to underflow)")

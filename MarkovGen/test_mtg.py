"""Markov Text Generator.

Patrick Wang, 2023

Resources:
Jelinek 1985 "Markov Source Modeling of Text Generation"
"""

import nltk

from mtg import finish_sentence


def test_generator():
    """Test Markov text generator."""
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())

    words = finish_sentence(
        ["she", "was", "not"],
        1,
        corpus,
        randomize=False,
    )
    print(words)
    words1 = finish_sentence(["robot", "was", "not"], 3, corpus)
    # print(words1)
    # ['robot'], n=3:
    # ['robot', ',', 'and', 'the', 'two', 'miss', 'steeles', ',', 'as', 'she']
    # ['she', 'was', 'not'], n=1:
    # ['she', 'was', 'not', ',', ',', ',', ',', ',', ',', ',']
    # ['robot'], n=2:
    # ['robot', ',', 'and', 'the', 'same', 'time', ',', 'and', 'the', 'same']
    print(words)
    assert words == [
        "she",
        "was",
        "not",
        "in",
        "the",
        "world",
        ".",
    ] or words == [
        "she",
        "was",
        "not",
        "in",
        "the",
        "world",
        ",",
        "and",
        "the",
        "two",
    ]


if __name__ == "__main__":
    test_generator()

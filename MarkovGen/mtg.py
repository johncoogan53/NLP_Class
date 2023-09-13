"""This will be a simple Markox text generator
which will use the function finish_sentence(sentence, n, corpus, randomize = False)
args:
    sentence[list]: list of tokens we are trying to build on
    n [int]: length of n-grams to use for prediction
    corpus: source corpus [list of tokens]
    randomize: stochastic or deterministic subsequent word selection"""
import numpy as np


def finish_sentence(sentence, n, corpus, randomize=False):
    """This function will take the input sentence, and use an n-gram model with stupid
    backoff (alpha =1) to assign subsequent words until a '.','?','!' or 10 total tokens
    """
    final_sentence = ""

    # take a sentence of tokens and predict the next word in that sentence [list] of tokens based
    # on 'n' past words
    # take the corpus and make a vocabulary (only unique items)
    corpus_array = np.array(corpus)
    vocabulary = np.unique(corpus_array)
    print(corpus_array)
    sentence_array = np.array(sentence)

    # assign probabilities to each vocabulary word based off occurence within the corpus
    prob = np.empty_like(vocabulary)
    for i in vocabulary:
        p[i] = np.count_nonzero(corpus_array[i] == vocabulary[i])
    print(p)
    exit()
    return final_sentence


def main():
    corp = ["hello", "the", "the", "hello", "goodbye", "a", "?", "ok"]
    test_n = 2
    sent = ["goodbye", "hello"]
    finish_sentence(sent, test_n, corp)

    return None


if __name__ == "__main__":
    main()

"""This will be a simple Markox text generator
which will use the function finish_sentence(sentence, n, corpus, randomize = False)
args:
    sentence[list]: list of tokens we are trying to build on
    n [int]: length of n-grams to use for prediction
    corpus: source corpus [list of tokens]
    randomize: stochastic or deterministic subsequent word selection"""
from collections import defaultdict
import numpy as np


def n_gram_counter(corpus, n):
    """generates a dictionary of dictionaries for all n grams in the corpus with an n*len(corpus) efficiency"""
    ngram_counter = defaultdict(dict)
    for i in range(len(corpus)):
        for j in range(n + 1):
            ngram = tuple(corpus[i : i + j])
            ngram_counter[j][ngram] = ngram_counter[j].get(ngram, 0) + 1
    return dict(ngram_counter)


def compute_prob(ngram, prob_mat, corpus):
    """uses the counter matrix to determine the next best word"""
    probability = 0
    # recursive base case
    n = len(ngram)
    prior = ngram[:-1]
    if n == 1:
        probability = prob_mat[n][ngram] / len(corpus)
        return probability

    if ngram in prob_mat[n].keys():
        num = prob_mat[n][ngram]
        denom = prob_mat[n - 1][prior]
        probability = num / denom
    else:
        probability = 0.4 * compute_prob(ngram[1:], prob_mat, corpus)
    return probability


def finish_sentence(sentence, n, corpus, randomize=False):
    """this function will compute probabilities based off prior sentences and pick the best word"""
    v, ind = np.unique(np.array(corpus), return_index=True)
    vocab = v[np.argsort(ind)]
    final_sentence = np.array(sentence)
    vocabulary = vocab.tolist()

    count_matrix = n_gram_counter(corpus, n)
    # print(count_matrix[1])
    # print(f"Count of not be: {count_matrix[2][('not','be')]}")
    # print(f"Count of not: {count_matrix[1][('not',)]}")
    # print(f"Count of was not in: {count_matrix[3][('was', 'not', 'in')]}")
    # print(f"Count of was not: {count_matrix[2][('was', 'not')]}")

    best_word_indx = 0
    while (
        # run until we get to a sentence of 10 tokens or punctuation
        len(final_sentence) < 10
        and vocabulary[best_word_indx] != "."
        and vocabulary[best_word_indx] != "!"
        and vocabulary[best_word_indx] != "?"
    ):
        if n > len(final_sentence):
            prior = final_sentence
            pass
        else:
            prior = final_sentence[-(n - 1) :]
            pass
        best_word_indx = 0
        prob = 0
        equal_words = []
        # pylint: disable = Consider using enumerate instead of iterating with range and lenPylintC0200:consider-using-enumerate
        for i in range(len(vocabulary)):
            # append vocabulary words onto prior and compute the backoff probability

            curr_gram = tuple(np.append(prior, vocabulary[i]))
            if n == 1:
                curr_gram = (vocabulary[i],)
            # print(f"computing probability for {curr_gram} ")
            curr_prob = compute_prob(curr_gram, count_matrix, corpus)

            if curr_prob > prob:  # handles the deterministic case
                prob = curr_prob
                best_word_indx = i
                equal_words = [vocabulary[i]]
            elif curr_prob == prob and randomize is True:
                # create a list of words of equal probability to select from
                equal_words.append(vocabulary[i])
                pass
            else:
                pass
        if len(equal_words) == 0:  # no words of equal probability found
            final_sentence = np.append(final_sentence, vocabulary[best_word_indx])
        else:  # take a random selection from the words of equal probability
            eq_words = np.array(equal_words)
            word_chosen = np.random.choice(eq_words)
            final_sentence = np.append(final_sentence, word_chosen)
    return final_sentence


def main():
    corp = [
        "the",
        "cat",
        "is",
        "chubby",
        ",",
        "i",
        "want",
        "to",
        "pet",
        "his",
        "belly",
        ".",
        "in",
        "the",
        "cat",
        "is",
        "was",
        "not",
        "in",
        "the",
        "cat",
        "ate",
        "the",
        "dog",
        "the",
        "cat",
        "is",
        "huge",
        "and",
        "silly",
        "cat",
        "is",
        "fat",
        "the",
        "cat",
        "ran",
        "the",
        "cat",
        "is",
        "scared",
    ]
    corp1 = []
    test_n = 1
    sent = ["the", "cat", "is"]
    print(finish_sentence(sent, test_n, corp, randomize=True))

    return None


if __name__ == "__main__":
    main()

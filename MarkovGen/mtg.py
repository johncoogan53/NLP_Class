"""This will be a simple Markox text generator
which will use the function finish_sentence(sentence, n, corpus, randomize = False)
args:
    sentence[list]: list of tokens we are trying to build on
    n [int]: length of n-grams to use for prediction
    corpus: source corpus [list of tokens]
    randomize: stochastic or deterministic subsequent word selection"""


def finish_sentence(sentence, n, corpus, randomize=False):
    """This function will take the input sentence, and use an n-gram model with stupid
    backoff (alpha =1) to assign subsequent words until a '.','?','!' or 10 total tokens
    """
    final_sentence = ""

    return final_sentence

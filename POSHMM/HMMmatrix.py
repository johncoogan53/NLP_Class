"""Python file to create the HMM matrices for the viterbi algorithm:
obs: An iterable of ints representing observations.
pi: A 1D numpy array of floats representing initial state probabilities.
A: A 2D numpy array of floats representing state transition probabilities.
B: A 2D numpy array of floats representing emission probabilities."""
import numpy as np
import nltk
from collections import defaultdict


class HMMmatrix:
    """class for generating all viterbi arguments for the HMM"""

    def __init__(self, training_data):
        unique_second_parts = set()
        # Iterate through the list of lists
        for sentence in training_data:
            for tup in sentence:
                # Add the second part of the tuple to the set
                unique_second_parts.add(tup[1])

        # Count the number of unique second parts
        num_states = len(unique_second_parts)
        self.first_word_counts = defaultdict(int)
        self.training_data = training_data
        for sentence in self.training_data:
            first_word_pos = sentence[0][1]
            self.first_word_counts[first_word_pos] += 1

        self.num_states = num_states
        self.unique_words = {}
        self.pos_to_index = {
            pos: i for i, pos in enumerate(self.first_word_counts.keys())
        }

    def create_pi(self):
        """Creates the initial state probabilities for the HMM.
        Args:
            num_states: The number of states in the HMM.
        Returns:
            A 1D numpy array of floats representing initial state probabilities.
        """
        pi = np.zeros((self.num_states,))
        pi = np.array(list(self.first_word_counts.values())) / len(self.training_data)
        return pi

    def trans_matrix(self):
        """Creates the transition matrix for the HMM."""
        transition_matrix = np.ones((self.num_states, self.num_states))
        for sentence in self.training_data:
            for i in range(len(sentence) - 1):
                pos1 = sentence[i][1]
                pos2 = sentence[i + 1][1]
                transition_matrix[self.pos_to_index[pos1]][self.pos_to_index[pos2]] += 1

        transition_matrix = transition_matrix / transition_matrix.sum(
            axis=1, keepdims=True
        )
        return transition_matrix

    def emission_matrix(self):
        """Creates the emission matrix for the HMM."""
        # generate a dictionary of unique words in the training data.

        idx = 0
        for sentence in self.training_data:
            for word, pos in sentence:
                if word not in self.unique_words:
                    self.unique_words[word] = idx
                    idx += 1
                else:
                    continue

        em_matrix = np.ones((self.num_states, len(self.unique_words)+1))
        for sentence in self.training_data:
            for word, pos in sentence:
                em_matrix[self.pos_to_index[pos]][self.unique_words[word]] += 1

        emission_matrix = em_matrix / em_matrix.sum(axis=1, keepdims=True)
        return emission_matrix


def main():
    training_data = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]

    first_words = HMMmatrix(training_data)
    check_UNK = first_words.emission_matrix()
    print(check_UNK[-1])
    return None


if __name__ == "__main__":
    main()

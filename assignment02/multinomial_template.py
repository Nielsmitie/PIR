#! /usr/bin/python
# -*- coding: utf-8 -*-


"""MLE for the multinomial distribution."""


from argparse import ArgumentParser
import math


def get_words(file_path):
    """Return a list of words from a file, converted to lower case."""
    with open(file_path, encoding='utf-8') as hfile:
        return hfile.read().lower().split()


def get_probabilities(words, stopwords, k):
    """
    Create a multinomial probability distribution from a list of words:
        1. Find the top-k most frequent words.
        2. For every one of the most frequent words, calculate its probability according to MLE.

    Return a dictionary of size k that maps the words to their probabilities.
    """

    # determine the k most frequent words
    words_dic = {}

    for word in words:
        if word not in stopwords:
            if word not in words_dic:
                words_dic[word] = 0
            words_dic[word] += 1

    import operator
    sorted_x = sorted(words_dic.items(), key=operator.itemgetter(1), reverse=True)

    sorted_tuple = sorted_x[:k]
    top_k_words = [x[0] for x in sorted_tuple]
    # use the formula from problem 3 the MLE of a multinomial distribution to estimate the probability of each word
    # Formula: p[i] = x[i] / n
    # p[i] probability, x[i] number of occurrences, n sum of number of occurrences
    # calculate n
    sum_of_occurrences = 0
    for word in top_k_words:
        sum_of_occurrences += words_dic[word]

    # calculate the probability
    word_probs = dict(zip(top_k_words, [0]*len(top_k_words)))
    for word in top_k_words:
        word_probs[word] = words_dic[word] / sum_of_occurrences

    return word_probs


def multinomial_pmf(sample, probabilities):
    """
    The multinomial probability mass function.
    Inputs:
        * sample: dictionary, maps words (X_i) to observed frequencies (x_i)
        * probabilities: dictionary, maps words to their probabilities (p_i)

    Return the probability of observing the sample, i.e. P(X_1=x_1, ..., X_k=x_k).
    """
    sum_of_all_samples = 0
    ratios = 1
    for key in sample.keys():
        sum_of_all_samples += sample[key]
        ratios *= math.pow(probabilities[key], sample[key]) / math.factorial(sample[key])

    return math.factorial(sum_of_all_samples) * ratios


def main():
    '''
    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUT_FILE', help='A file containing whitespace-delimited words')
    arg_parser.add_argument('SW_FILE', help='A file containing whitespace-delimited stopwords')
    arg_parser.add_argument('-k', type=int, default=10,
                            help='How many of the most frequent words to consider')
    args = arg_parser.parse_args()


    words = get_words(args.INPUT_FILE)
    stopwords = set(get_words(args.SW_FILE))
    probabilities = get_probabilities(words, stopwords, args.k)
    '''
    input_file = "shakespeare.txt"
    stop_words = "stopwords.txt"
    k = 5
    words = get_words(input_file)
    stopwords = set(get_words(stop_words))
    probabilities = get_probabilities(words, stopwords, k)

    # we should have k probabilities
    # assert len(probabilities) == args.k
    assert len(probabilities) == k

    # check if all p_i sum to 1 (accounting for some rounding error)
    assert 1 - 1e-12 <= sum(probabilities.values()) <= 1 + 1e-12

    # check if p_i >= 0
    assert not any(p < 0 for p in probabilities.values())

    # print estimated probabilities
    print('estimated probabilities:')
    i = 1
    for word, prob in probabilities.items():
        print('p_{}\t{}\t{:.5f}'.format(i, word, prob))
        i += 1

    # read inputs for x_i
    print('\nenter sample:')
    sample = {}
    i = 1
    for word in probabilities:
        sample[word] = int(input('X_{}='.format(i)))
        i += 1

    # print P(X_1=x_1, ..., X_k=x_k)
    print('\nresult: {}'.format(multinomial_pmf(sample, probabilities)))


if __name__ == '__main__':
    main()

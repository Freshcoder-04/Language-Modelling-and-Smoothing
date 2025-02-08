import os
import sys
import random
from collections import defaultdict, Counter
import math
from tokenizer import custom_nlp_tokenizer
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import itertools
from nltk.util import ngrams


def pad_sentence(sentence, k, start_token="<s>"):
    padded_sentence = [start_token] * (k) + sentence
    return padded_sentence


def generate_k_gram_models_with_counts(corpus_path, n):
    """
    Generates k-gram models and (k-1)-gram counts for all k from 1 to n.

    Args:
        corpus_path (str): Path to the corpus file.
        n (int): The maximum value of N for the N-gram models.
        tokenizer (function): The tokenizer function to preprocess and tokenize the text.

    Returns:
        tuple: Two lists and vocabulary size:
            - n_gram_counts_list: A list of dictionaries with k-gram counts for k from 1 to n.
            - n_minus_1_gram_counts_list: A list of dictionaries with (k-1)-gram counts for k from 1 to n.
            - vocabulary_size: The size of the vocabulary in the corpus.
    """
    n_gram_counts_list = []
    n_minus_1_gram_counts_list = []
    vocabulary = set()

    # Read the corpus
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"The file at {corpus_path} does not exist.")

    with open(corpus_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the text using the provided tokenizer
    tokenized_sentences = custom_nlp_tokenizer(text)

    # Generate k-gram counts for all k from 1 to n
    for k in range(1, n + 1):
        n_gram_counts = defaultdict(int)
        n_minus_1_gram_counts = defaultdict(int)

        for sentence in tokenized_sentences:
            n_grams = list(ngrams(sentence, k))
            n_minus_1_grams = list(ngrams(sentence, k - 1)) if k > 1 else []

            for n_gram in n_grams:
                n_gram_counts[n_gram] += 1
                vocabulary.update(n_gram)  # Add tokens to the vocabulary

            for n_minus_1_gram in n_minus_1_grams:
                n_minus_1_gram_counts[n_minus_1_gram] += 1

        n_gram_counts_list.append(n_gram_counts)
        n_minus_1_gram_counts_list.append(n_minus_1_gram_counts)

    vocabulary_size = len(vocabulary)  # Size of the unique vocabulary
    return n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size


def laplace_conditional_probability(n_gram, n_minus_1_gram, n_gram_counts, n_minus_1_gram_counts, vocabulary_size):
    """
    Calculates the conditional probability of an N-gram using Laplace smoothing.

    Args:
        n_gram (tuple): The N-gram.
        n_minus_1_gram (tuple): The (N-1)-gram.
        n_gram_counts (dict): Dictionary with N-gram counts.
        n_minus_1_gram_counts (dict): Dictionary with (N-1)-gram counts.
        vocabulary_size (int): Size of the vocabulary.

    Returns:
        float: The conditional probability of the N-gram.
    """
    n_gram_count = n_gram_counts.get(n_gram, 0)
    n_minus_1_gram_count = n_minus_1_gram_counts.get(n_minus_1_gram, 0)
    return (n_gram_count + 1) / (n_minus_1_gram_count + vocabulary_size)


def good_turing_conditional_probability(n_gram, n_minus_1_gram, n_gram_counts, n_minus_1_gram_counts, vocabulary_size, count_of_counts, plot=False):
    """
    Calculates the conditional probability of an N-gram using Good-Turing smoothing.

    Args:
        n_gram (tuple): The N-gram.
        n_minus_1_gram (tuple): The (N-1)-gram.
        n_gram_counts (dict): Dictionary with N-gram counts.
        n_minus_1_gram_counts (dict): Dictionary with (N-1)-gram counts.
        vocabulary_size (int): Size of the vocabulary.
        count_of_counts (dict): Dictionary with count of counts (N_r).
        plot (bool): If True, plot the Nr vs r curve in log-log scale and fit a line.

    Returns:
        float: The conditional probability of the N-gram.
    """
    # Step 1: Prepare data for plotting (Nr vs r)
    r_values = sorted(count_of_counts.keys())
    Nr_values = [count_of_counts[r] for r in r_values]

    # Step 2: Compute log(r) and log(Nr)
    log_r_values = np.log(r_values)
    log_Nr_values = np.log(Nr_values)

    # Step 3: Identify the cutoff where the slope becomes close to zero
    slopes = []
    for i in range(1, len(log_r_values)):
        slope = (log_Nr_values[i] - log_Nr_values[i - 1]) / (log_r_values[i] - log_r_values[i - 1])
        slopes.append(slope)

    # Find the cutoff where the slope becomes close to zero
    cutoff_index = next((i for i, slope in enumerate(slopes) if abs(slope) < 0.1), len(slopes))
    cutoff_r = r_values[cutoff_index]

    # Step 4: Fit a linear regression line only to the data before the cutoff
    log_r_before_cutoff = log_r_values[:cutoff_index]
    log_Nr_before_cutoff = log_Nr_values[:cutoff_index]

    slope, intercept, r_value, p_value, std_err = linregress(log_r_before_cutoff, log_Nr_before_cutoff)
    regression_line = slope * log_r_values + intercept

    # Step 5: Predict Nr values after the cutoff using the regression line
    smoothed_count_of_counts = {}
    for r, Nr in zip(r_values, Nr_values):
        if r >= cutoff_r:
            # Use the regression line to predict Nr for r >= cutoff_r
            smoothed_count_of_counts[r] = np.exp(slope * np.log(r) + intercept)
        else:
            # Use the original Nr for r < cutoff_r
            smoothed_count_of_counts[r] = Nr

    # Step 6: Estimate N_0 (number of unseen n-grams)
    total_possible_ngrams = vocabulary_size ** len(n_gram)
    observed_ngrams = len(n_gram_counts)
    smoothed_count_of_counts[0] = max(total_possible_ngrams - observed_ngrams, 1)  # Ensure N_0 is at least 1

    # Step 7: Calculate Good-Turing adjusted counts
    def good_turing_adjusted_count(r):
        if r == 0:
            return smoothed_count_of_counts.get(1, 1) / smoothed_count_of_counts.get(0, 1)
        if r + 1 not in smoothed_count_of_counts or r not in smoothed_count_of_counts:
            return r  # Fallback to original count if N_{r+1} or N_r is not available
        return (r + 1) * (smoothed_count_of_counts[r + 1] / smoothed_count_of_counts[r])

    n_gram_count = n_gram_counts.get(n_gram, 0)
    n_minus_1_gram_count = n_minus_1_gram_counts.get(n_minus_1_gram, 0)

    adjusted_n_gram_count = good_turing_adjusted_count(n_gram_count)
    adjusted_n_minus_1_gram_count = good_turing_adjusted_count(n_minus_1_gram_count)

    # Step 8: Calculate probability
    if adjusted_n_minus_1_gram_count == 0:
        probability = 1 / vocabulary_size  # Fallback for zero denominator
    else:
        probability = adjusted_n_gram_count / adjusted_n_minus_1_gram_count

    # Step 9: Plot the data, regression line, and cutoff line (if requested)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(log_r_values, log_Nr_values, color='blue', label='Data Points')
        plt.plot(log_r_values, regression_line, color='red', label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')
        plt.axvline(x=np.log(cutoff_r), color='green', linestyle='--', label=f'Cutoff: log(r) = {np.log(cutoff_r):.2f}')
        plt.xlabel('log(r)')
        plt.ylabel('log(Nr)')
        plt.title('Good-Turing: Nr vs r in log-log scale')
        plt.legend()
        plt.grid(True)
        plt.show()

    return probability


def linear_interpolation_conditional_probability(n_gram, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, lambdas):
    """
    Calculates the conditional probability of an N-gram using linear interpolation.

    Args:
        n_gram (tuple): The N-gram.
        n_gram_counts_list (list): List of dictionaries with k-gram counts for k from 1 to n.
        n_minus_1_gram_counts_list (list): List of dictionaries with (k-1)-gram counts for k from 1 to n.
        vocabulary_size (int): Size of the vocabulary.
        lambdas (list): List of weights for each k-gram model (e.g., [λ1, λ2, λ3]).

    Returns:
        float: The conditional probability of the N-gram.
    """
    interpolated_probability = 0.0
    for k in range(len(lambdas)):
        n_gram_k = n_gram[k:]  # Use (k+1)-gram
        n_minus_1_gram_k = n_gram_k[:-1] if len(n_gram_k) > 1 else tuple()

        # Get counts
        n_gram_count = n_gram_counts_list[k].get(n_gram_k, 0)
        n_minus_1_gram_count = n_minus_1_gram_counts_list[k].get(n_minus_1_gram_k, 0)

        # Calculate probability for the (k+1)-gram model
        if k == 0:  # Unigram model
            probability = (n_gram_count + 1) / (sum(n_gram_counts_list[k].values()) + vocabulary_size)
        else:  # Bigram, trigram, etc.
            probability = (n_gram_count + 1) / (n_minus_1_gram_count + vocabulary_size)

        # Add weighted probability to the interpolated probability
        interpolated_probability += lambdas[k] * probability

    return interpolated_probability


def calculate_sentence_probability(sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method, lambdas=None, plot=False):
    """
    Calculates the probability of a sentence using the specified smoothing method.

    Args:
        sentence (str): The input sentence.
        n (int): The value of N for the N-gram model.
        n_gram_counts_list (list): List of dictionaries with k-gram counts for k from 1 to n.
        n_minus_1_gram_counts_list (list): List of dictionaries with (k-1)-gram counts for k from 1 to n.
        vocabulary_size (int): Size of the vocabulary.
        smoothing_method (str): The smoothing method to use ('laplace', 'good_turing', 'interpolation').
        lambdas (list): List of weights for each k-gram model (required for interpolation).
        plot (bool): If True, plot the Nr vs r curve in log-log scale and fit a line (for Good-Turing).

    Returns:
        tuple: The probability and perplexity of the sentence.
    """
    tokenized_sentence = custom_nlp_tokenizer(sentence)[0]
    if len(tokenized_sentence) < n:
        tokenized_sentence = pad_sentence(tokenized_sentence, n - len(tokenized_sentence))
    if len(tokenized_sentence) < n:
        return 0.0, math.inf

    log_probability = 0.0
    log_probabilities = []

    if smoothing_method == 'good_turing':
        # Calculate count of counts (N_r) for Good-Turing
        count_of_counts = defaultdict(int)
        for n_gram, count in n_gram_counts_list[-1].items():
            count_of_counts[count] += 1

    for i in range(len(tokenized_sentence) - n + 1):
        n_gram = tuple(tokenized_sentence[i:i + n])
        n_minus_1_gram = n_gram[:-1]

        if smoothing_method == 'laplace':
            conditional_probability = laplace_conditional_probability(n_gram, n_minus_1_gram, n_gram_counts_list[-1], n_minus_1_gram_counts_list[-1], vocabulary_size)
        elif smoothing_method == 'good_turing':
            conditional_probability = good_turing_conditional_probability(n_gram, n_minus_1_gram, n_gram_counts_list[-1], n_minus_1_gram_counts_list[-1], vocabulary_size, count_of_counts, plot)
        elif smoothing_method == 'interpolation':
            conditional_probability = linear_interpolation_conditional_probability(n_gram, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, lambdas)
        else:
            raise ValueError("Invalid smoothing method. Choose 'laplace', 'good_turing', or 'interpolation'.")

        log_probabilities.append(math.log(conditional_probability + 1e-10))  # Avoid log(0)

    log_probability = sum(log_probabilities)
    SP = math.exp(log_probability)
    PP = math.exp(-1 / len(log_probabilities) * log_probability)
    return SP, PP


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 language_model.py <lm_type> <corpus_path>")
        sys.exit(1)

    lm_type = sys.argv[1].lower()
    corpus_path = sys.argv[2]

    if lm_type not in ['l', 'g', 'i']:
        print("Invalid LM type. Choose 'l' for Laplace, 'g' for Good-Turing, or 'i' for Interpolation.")
        sys.exit(1)

    # Set N for N-grams (e.g., trigrams)
    n = 3

    # Generate k-gram models for all k from 1 to n
    n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size = generate_k_gram_models_with_counts(corpus_path, n)

    # Define weights for interpolation (e.g., λ1, λ2, λ3)
    lambdas = [0.4,0.3,0.3]
    print(lambdas)

    while True:
        input_sentence = input("Input sentence: ").strip()
        if not input_sentence:
            print("Exiting...")
            break

        if lm_type == 'l':
            P, PP = calculate_sentence_probability(input_sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method='laplace')
            print(f"Score: {PP}, Probability: {P}")
        elif lm_type == 'g':
            P, PP = calculate_sentence_probability(input_sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method='good_turing', plot=True)
            print(f"Score: {PP}, Probability: {P}")
        elif lm_type == 'i':
            P, PP = calculate_sentence_probability(input_sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method='interpolation', lambdas=lambdas)
            print(f"Score: {PP}, Probability: {P}")
        else:
            print("Only Laplace (l), Good-Turing (g), and Interpolation (i) smoothing are implemented in this version.")
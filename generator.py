import sys
from collections import defaultdict
import itertools
import numpy as np
from scipy.stats import linregress
from language_model import (
    generate_k_gram_models_with_counts,
    laplace_conditional_probability,
    good_turing_conditional_probability,
    linear_interpolation_conditional_probability,
    pad_sentence,
    custom_nlp_tokenizer,
)

def get_top_k_next_words(sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method, k, lambdas=None):
    """
    Returns the top k most probable next words for a given sentence.

    Args:
        sentence (str): The input sentence.
        n (int): The value of N for the N-gram model.
        n_gram_counts_list (list): List of dictionaries with k-gram counts for k from 1 to n.
        n_minus_1_gram_counts_list (list): List of dictionaries with (k-1)-gram counts for k from 1 to n.
        vocabulary_size (int): Size of the vocabulary.
        smoothing_method (str): The smoothing method to use ('laplace', 'good_turing', 'interpolation', 'none').
        k (int): The number of candidate next words to return.
        lambdas (list): List of weights for each k-gram model (required for interpolation).

    Returns:
        list: A list of tuples containing the top k next words and their probabilities.
    """
    # Tokenize the input sentence
    tokenized_sentence = custom_nlp_tokenizer(sentence)[0]
    tokenized_sentence = tokenized_sentence[:-1]
    if len(tokenized_sentence) < n - 1:
        tokenized_sentence = pad_sentence(tokenized_sentence, n - 1 - len(tokenized_sentence))

    # Get the last (n-1) tokens as the context
    context = tuple(tokenized_sentence[-(n - 1):])

    # Precompute count_of_counts and regression parameters for Good-Turing smoothing
    count_of_counts = defaultdict(int)
    regression_params = None
    if smoothing_method == 'good_turing':
        # Calculate count of counts (N_r)
        for n_gram_, count in n_gram_counts_list[-1].items():
            count_of_counts[count] += 1

        # Prepare data for regression
        r_values = sorted(count_of_counts.keys())
        Nr_values = [count_of_counts[r] for r in r_values]
        log_r_values = np.log(r_values)
        log_Nr_values = np.log(Nr_values)

        # Identify the cutoff where the slope becomes close to zero
        slopes = []
        for i in range(1, len(log_r_values)):
            slope = (log_Nr_values[i] - log_Nr_values[i - 1]) / (log_r_values[i] - log_r_values[i - 1])
            slopes.append(slope)

        cutoff_index = next((i for i, slope in enumerate(slopes) if abs(slope) < 0.1), len(slopes))
        cutoff_r = r_values[cutoff_index]

        # Fit a linear regression line only to the data before the cutoff
        log_r_before_cutoff = log_r_values[:cutoff_index]
        log_Nr_before_cutoff = log_Nr_values[:cutoff_index]
        slope, intercept, r_value, p_value, std_err = linregress(log_r_before_cutoff, log_Nr_before_cutoff)

        # Store regression parameters
        regression_params = (slope, intercept, cutoff_r)

    # Calculate the conditional probabilities for all possible next tokens
    next_word_probs = []
    for token in set(itertools.chain.from_iterable(n_gram_counts_list[-1].keys())):
        n_gram = context + (token,)
        n_minus_1_gram = context

        if smoothing_method == 'laplace':
            prob = laplace_conditional_probability(n_gram, n_minus_1_gram, n_gram_counts_list[-1], n_minus_1_gram_counts_list[-1], vocabulary_size)
        elif smoothing_method == 'good_turing':
            prob = good_turing_conditional_probability(n_gram, n_minus_1_gram, n_gram_counts_list[-1], n_minus_1_gram_counts_list[-1], vocabulary_size, count_of_counts, regression_params)
        elif smoothing_method == 'interpolation':
            prob = linear_interpolation_conditional_probability(n_gram, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, lambdas)
        elif smoothing_method == 'none':
            # Raw probability without smoothing
            prob = n_gram_counts_list[-1].get(n_gram, 0) / n_minus_1_gram_counts_list[-1].get(n_minus_1_gram, 1)
        else:
            raise ValueError("Invalid smoothing method. Choose 'laplace', 'good_turing', 'interpolation', or 'none'.")

        next_word_probs.append((token, prob))

    # Sort by probability in descending order
    next_word_probs.sort(key=lambda x: x[1], reverse=True)

    # Return the top k candidates
    return next_word_probs[:k]

def generate_sequence(starting_sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method, sequence_length, lambdas=None):
    """
    Generates a sequence of words using the N-gram model.

    Args:
        starting_sentence (str): The initial sentence to begin the generation.
        n (int): The value of N for the N-gram model.
        n_gram_counts_list (list): List of dictionaries with k-gram counts for k from 1 to n.
        n_minus_1_gram_counts_list (list): List of dictionaries with (k-1)-gram counts for k from 1 to n.
        vocabulary_size (int): Size of the vocabulary.
        smoothing_method (str): The smoothing method to use ('laplace', 'good_turing', 'interpolation', 'none').
        sequence_length (int): The desired length of the generated sequence.
        lambdas (list): List of weights for each k-gram model (required for interpolation).

    Returns:
        str: The generated sequence.
    """
    current_sentence = starting_sentence

    for _ in range(sequence_length):
        top_next_words = get_top_k_next_words(current_sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method, k=1, lambdas=lambdas)
        if not top_next_words or top_next_words[0][1] == 0:
            break

        # Append the most probable next word to the sentence
        next_word = top_next_words[0][0]
        current_sentence += f" {next_word}"

    return current_sentence

if __name__ == "__main__":
    while True:
        if len(sys.argv) < 4:
            print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
            sys.exit(1)

        lm_type = sys.argv[1].lower()
        corpus_path = sys.argv[2]
        k = int(sys.argv[3])

        if lm_type not in ['l', 'g', 'i', 'n']:
            print("Invalid LM type. Choose 'l' for Laplace, 'g' for Good-Turing, 'i' for Interpolation, or 'n' for None (raw N-gram).")
            sys.exit(1)

        # Set N for N-grams (e.g., trigrams)
        n = 3

        # Generate k-gram models for all k from 1 to n
        n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size = generate_k_gram_models_with_counts(corpus_path, n)

        # Define weights for interpolation (e.g., λ1, λ2, λ3)
        lambdas = [0.3, 0.4, 0.3]

        # Determine the smoothing method
        smoothing_method = 'laplace' if lm_type == 'l' else 'good_turing' if lm_type == 'g' else 'interpolation' if lm_type == 'i' else 'none'

        # Prompt the user for a sentence
        input_sentence = input("Input sentence: ").strip()

        # Prompt the user for sequence generation
        generate_seq = input("Generate a sequence? (y/n): ").strip().lower() == 'y'
        
        if generate_seq:
            sequence_length = int(input("Enter the desired sequence length: ").strip())
            generated_sequence = generate_sequence(input_sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method, sequence_length, lambdas)
            print(f"Generated Sequence: {generated_sequence}")
        else:
            # Get the top k next words and their probabilities
            top_k_next_words = get_top_k_next_words(input_sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method, k, lambdas)

            # Print the results
            for word, prob in top_k_next_words:
                print(f"{word} {prob:.4f}")
    
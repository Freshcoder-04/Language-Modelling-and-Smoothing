import os
import random
from collections import defaultdict
import numpy as np
from language_model import (
    generate_k_gram_models_with_counts,
    laplace_conditional_probability,
    good_turing_conditional_probability,
    linear_interpolation_conditional_probability,
    pad_sentence,
    custom_nlp_tokenizer,
)


# Step 1: Split the corpus into train and test sets
def split_corpus(corpus_path, train_ratio=0.8):
    """
    Splits the corpus into train and test sets.

    Args:
        corpus_path (str): Path to the corpus file.
        train_ratio (float): Ratio of the corpus to use for training.

    Returns:
        tuple: (train_text, test_text)
    """
    with open(corpus_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into sentences
    sentences = text.split('.')  # Simple split by period (can be improved)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences

    # Shuffle and split into train and test
    random.shuffle(sentences)
    split_index = int(len(sentences) * train_ratio)
    train_sentences = sentences[:split_index]
    test_sentences = sentences[split_index:]

    return train_sentences, test_sentences


# Step 2: Save train and test sets to the interim folder
def save_train_test_sets(corpus_name, train_sentences, test_sentences):
    """
    Saves the train and test sets to the interim folder.

    Args:
        corpus_name (str): Name of the corpus.
        train_sentences (list): List of training sentences.
        test_sentences (list): List of test sentences.
    """
    interim_folder = os.path.join("corpora", "interim", corpus_name)
    os.makedirs(interim_folder, exist_ok=True)

    # Save train set
    with open(os.path.join(interim_folder, "train.txt"), 'w', encoding='utf-8') as file:
        file.write(". ".join(train_sentences) + ".")

    # Save test set
    with open(os.path.join(interim_folder, "test.txt"), 'w', encoding='utf-8') as file:
        file.write(". ".join(test_sentences) + ".")


# Step 3: Compute perplexity for each sentence
def compute_perplexity(sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method, lambdas=None):
    """
    Computes the perplexity of a sentence using the specified smoothing method.

    Args:
        sentence (str): The input sentence.
        n (int): The value of N for the N-gram model.
        n_gram_counts_list (list): List of dictionaries with k-gram counts for k from 1 to n.
        n_minus_1_gram_counts_list (list): List of dictionaries with (k-1)-gram counts for k from 1 to n.
        vocabulary_size (int): Size of the vocabulary.
        smoothing_method (str): The smoothing method to use ('laplace', 'good_turing', 'interpolation').
        lambdas (list): List of weights for each k-gram model (required for interpolation).

    Returns:
        float: The perplexity of the sentence.
    """
    tokenized_sentence = custom_nlp_tokenizer(sentence)[0]
    if len(tokenized_sentence) < n:
        tokenized_sentence = pad_sentence(tokenized_sentence, n - len(tokenized_sentence))
    if len(tokenized_sentence) < n:
        return float('inf')  # Handle short sentences

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
            conditional_probability = good_turing_conditional_probability(n_gram, n_minus_1_gram, n_gram_counts_list[-1], n_minus_1_gram_counts_list[-1], vocabulary_size, count_of_counts)
        elif smoothing_method == 'interpolation':
            conditional_probability = linear_interpolation_conditional_probability(n_gram, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, lambdas)
        else:
            raise ValueError("Invalid smoothing method. Choose 'laplace', 'good_turing', or 'interpolation'.")

        log_probabilities.append(np.log(conditional_probability + 1e-10))  # Avoid log(0)

    log_probability = np.sum(log_probabilities)
    perplexity = np.exp(-log_probability / len(log_probabilities)) if len(log_probabilities) > 0 else float('inf')
    return perplexity


# Step 4: Write perplexity scores to a file
def write_perplexity_scores(output_file, sentences, perplexities):
    """
    Writes perplexity scores to a file in the specified format.

    Args:
        output_file (str): Path to the output file.
        sentences (list): List of sentences.
        perplexities (list): List of perplexity scores.
    """
    avg_perplexity = np.mean(perplexities)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"{avg_perplexity:.4f}\n")
        for sentence, perplexity in zip(sentences, perplexities):
            file.write(f"{sentence}\t{perplexity:.4f}\n")


# Step 5: Main function to process corpora
def process_corpora(corpora_folder, roll_number):
    """
    Processes the corpora and computes perplexity scores.

    Args:
        corpora_folder (str): Path to the folder containing the corpora.
        roll_number (str): Roll number for naming output files.
    """
    corpora = ["Pride and Prejudice - Jane Austen", "Ulysses - James Joyce"]
    smoothing_methods = ['laplace', 'good_turing', 'interpolation']
    lambdas = [0.3, 0.4, 0.3]  # Weights for interpolation

    for corpus_name in corpora:
        corpus_path = os.path.join(corpora_folder, "external", f"{corpus_name}.txt")
        train_sentences, test_sentences = split_corpus(corpus_path)
        save_train_test_sets(corpus_name, train_sentences, test_sentences)

        # Train the language model on the training set
        train_path = os.path.join("corpora", "interim", corpus_name, "train.txt")

        for n in [1, 3, 5]:  # N-gram models
            n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size = generate_k_gram_models_with_counts(train_path, n)

            for smoothing_method in smoothing_methods:
                # Skip Linear Interpolation for N=1
                if n == 1 and smoothing_method == 'interpolation':
                    continue

                # Compute perplexity for each sentence in the test set
                perplexities = []
                for sentence in test_sentences:
                    perplexity = compute_perplexity(sentence, n, n_gram_counts_list, n_minus_1_gram_counts_list, vocabulary_size, smoothing_method, lambdas)
                    perplexities.append(perplexity)

                # Determine LM type based on corpus and smoothing method
                if corpus_name == "Pride and Prejudice - Jane Austen":
                    if smoothing_method == 'laplace':
                        lm_type = "LM1"
                    elif smoothing_method == 'good_turing':
                        lm_type = "LM2"
                    elif smoothing_method == 'interpolation':
                        lm_type = "LM3"
                elif corpus_name == "Ulysses - James Joyce":
                    if smoothing_method == 'laplace':
                        lm_type = "LM4"
                    elif smoothing_method == 'good_turing':
                        lm_type = "LM5"
                    elif smoothing_method == 'interpolation':
                        lm_type = "LM6"

                # Write perplexity scores to the output file
                output_file = f"text_files/{roll_number}_{lm_type}_{n}_test-perplexity.txt"
                write_perplexity_scores(output_file, test_sentences, perplexities)


if __name__ == "__main__":
    corpora_folder = "corpora"
    roll_number = "2022102025"
    process_corpora(corpora_folder, roll_number)
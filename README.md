# N-gram Language Models

This repository contains code for building and evaluating N-gram language models with different smoothing techniques. The code is divided into three main files:

- **`language_model.py`**: Implements N-gram language models with Laplace smoothing, Good-Turing smoothing, and Linear Interpolation.
- **`generator.py`**: Generates the top-k most probable next words for a given sentence using the trained language models.
- **`tokenizer.py`**: Preprocesses and tokenizes text for use in the language models.

---

## Table of Contents

- [Requirements](#requirements)
- [File Descriptions](#file-descriptions)
- [How to Execute](#how-to-execute)
  - [Running `language_model.py`](#1-running-language_modelpy)
  - [Running `generator.py`](#2-running-generatorpy)
  - [Running `tokenizer.py`](#3-running-tokenizerpy)
- [Output Format](#output-format)
  - [Output for `language_model.py`](#1-language_modelpy)
  - [Output for `generator.py`](#2-generatorpy)
  - [Perplexity Scores for Test Sentences](#3-perplexity-scores-for-test-sentences)
- [Additional Information](#additional-information)
  - [Smoothing Techniques](#smoothing-techniques)
  - [File Naming Convention](#file-naming-convention)
  - [Notes](#notes)

---

## Requirements

To run the code, you need the following Python packages:

- `nltk`
- `numpy`
- `scipy`
- `matplotlib`

You can install the required packages using:

```bash
pip install nltk numpy scipy matplotlib
```

Additionally, download the necessary NLTK data:

```python
import nltk
nltk.download('punkt')
```

---

## File Descriptions

### 1. `language_model.py`

This file implements the following:

- **N-gram Language Models**: Generates N-gram and (N-1)-gram counts from a corpus.
- **Smoothing Techniques**:
  - **Laplace Smoothing**: Adds 1 to all counts to handle zero probabilities.
  - **Good-Turing Smoothing**: Adjusts counts using frequency of frequencies.
  - **Linear Interpolation**: Combines probabilities from unigram, bigram, and trigram models.
- **Perplexity Calculation**: Computes the perplexity of a sentence using the trained model.

### 2. `generator.py`

This file implements:

- **Next Word Prediction**: Given a sentence, it predicts the top-k most probable next words using the trained language model.
- **Good-Turing Smoothing Optimization**: Precomputes regression parameters for faster predictions.

### 3. `tokenizer.py`

This file implements:

- **Text Preprocessing**: Handles URLs, mentions, hashtags, percentages, ages, and time expressions.
- **Tokenization**: Tokenizes text into sentences and words using NLTK's `TreebankWordTokenizer`.

---

## How to Execute

### 1. Running `language_model.py`

To compute the probability and perplexity of a sentence using a specific smoothing method:

```bash
python3 language_model.py <lm_type> <corpus_path>
```

- `<lm_type>`: Specify the smoothing method:
  - `l` for Laplace Smoothing.
  - `g` for Good-Turing Smoothing.
  - `i` for Linear Interpolation.
- `<corpus_path>`: Path to the corpus file.

**Example:**

```bash
python3 language_model.py l corpora/external/Pride and Prejudice - Jane Austen.txt
```

### 2. Running `generator.py`

To generate the top-k most probable next words for a given sentence:

```bash
python3 generator.py <lm_type> <corpus_path> <k>
```

- `<lm_type>`: Specify the smoothing method (same as above).
- `<corpus_path>`: Path to the corpus file.
- `<k>`: Number of candidate next words to return.

**Example:**

```bash
python3 generator.py g corpora/external/Pride and Prejudice - Jane Austen.txt 3
```

### 3. Running `tokenizer.py`

To tokenize a given text:

```bash
python3 tokenizer.py
```

Enter the text when prompted, and the tokenized output will be displayed.

---

## Output Format

### 1. `language_model.py`

For each input sentence, the program outputs:

- **Probability**: The probability of the sentence.
- **Perplexity**: The perplexity of the sentence.

**Example:**

```
Input sentence: The cat sat on
Score: 45.6789, Probability: 0.1234
```

### 2. `generator.py`

For each input sentence, the program outputs the top-k most probable next words along with their probabilities.

**Example:**

```
Input sentence: The cat sat on
away 0.4000
happy 0.2000
fresh 0.1000
```

### 3. Perplexity Scores for Test Sentences

The perplexity scores for each sentence in the test set are saved in a text file with the following format:

```
avg_perplexity
sentence_1 <tab> perplexity
sentence_2 <tab> perplexity
...
```

**Example:**

```
45.6789
The cat sat on the mat.    50.1234
The dog chased the cat.    40.5678
```

---

## Additional Information

### Smoothing Techniques

- **Laplace Smoothing**:
  - Adds 1 to all counts to handle zero probabilities.
  - Suitable for small datasets.

- **Good-Turing Smoothing**:
  - Adjusts counts using frequency of frequencies.
  - Handles rare and unseen N-grams effectively.

- **Linear Interpolation**:
  - Combines probabilities from unigram, bigram, and trigram models.
  - Requires weights (lambdas) for each N-gram model.

### File Naming Convention

Perplexity scores are saved in files named:

```
<roll number>_LM1_N_test-perplexity.txt (Laplace Smoothing)
<roll number>_LM2_N_test-perplexity.txt (Good-Turing Smoothing)
<roll number>_LM3_N_test-perplexity.txt (Linear Interpolation)
```

Replace `<roll number>` with your actual roll number and `N` with the N-gram value (1, 3, or 5).

### Notes

- Ensure the corpus files are placed in the `corpora/external/` folder.
- The code skips Linear Interpolation for `N=1` since it doesn't make sense for unigrams.
- For Good-Turing smoothing, the regression line is fitted only to the linear region of the log-log plot.


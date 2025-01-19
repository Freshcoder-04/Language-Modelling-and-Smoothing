import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

# Download necessary NLTK data
nltk.download('punkt')

def custom_nlp_tokenizer(text):
    """
    Tokenizes the input text into sentences, and each sentence into words.
    Handles contractions (e.g., "don't" -> "do" + "not") and adds start/end tokens.
    
    Args:
        text (str): Input text to tokenize.

    Returns:
        list: A list of tokenized sentences, each prefixed with '<s>' and suffixed with '</s>'.
    """
    # Initialize tokenizer for splitting contractions
    tokenizer = TreebankWordTokenizer()
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize each sentence into words and handle contractions
    tokenized_sentences = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)  # Handles splitting contractions
        tokens = ['<s>'] + tokens + ['</s>']  # Add start and end tokens
        tokenized_sentences.append(tokens)
    
    return tokenized_sentences

# Example usage
text = "Don't stop believing. Isn't this amazing?"
result = custom_nlp_tokenizer(text)
print(result)
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text)
    text = re.sub(r"@\w+", "MENTION", text)
    text = re.sub(r"#\w+", "HASHTAG", text)
    text = re.sub(r"#\w+", "HASHTAG", text)
    text = re.sub(r"\b\d+(\.\d+)?\s?%", "PERCENTAGE", text)
    text = re.sub(r"\b\d+\s?(years old|yrs old|yo|years|yrs)\b", "AGE", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?\b", "TIME", text)
    text = re.sub(r"\b\d+\s?(seconds|minutes|hours|days|weeks|months|years)\b", 
                  "TIME_PERIOD", text, flags=re.IGNORECASE)
    return text

def custom_nlp_tokenizer(text):
    text = preprocess_text(text)
    
    tokenizer = TreebankWordTokenizer()

    sentences = sent_tokenize(text)
    
    tokenized_sentences = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokens = ['<s>'] + tokens + ['</s>']
        tokenized_sentences.append(tokens)
    
    return tokenized_sentences

if __name__ == '__main__':
    text = input("Your text: ")
    result = custom_nlp_tokenizer(text)
    print("Tokenized text:", result)
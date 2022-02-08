import numpy as np
import nltk
nltk.download('wordnet')
def my_bag_of_words(text,words_to_index, dict_size):
    """
    text: a string
    dict_size: size of the dictionary

    return a vector which is a bag-of-words representation of 'text'
"""
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    tokenText = tokenizer.tokenize(text)
    result_vector = np.zeros(dict_size)
    for word in tokenText:
        if word in words_to_index:
            idx = words_to_index[word]
            result_vector[idx] += 1
    return result_vector
            
            
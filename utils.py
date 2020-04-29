import re
from nltk.stem import WordNetLemmatizer
from string import punctuation as spcl_chars
import pickle


def get_filtered_words(words, test=False, pattern = r"[^a-zA-z0-9\s]"):
#     ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    
    return list(lemmatizer.lemmatize(re.sub(pattern, '', word)) for word in filter(lambda item: item not in spcl_chars, words) if re.sub(pattern, '', word)!="")


def word_feats(words):
    return dict([(word, True) for word in words])

def get_classifier(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
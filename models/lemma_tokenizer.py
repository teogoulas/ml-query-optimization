import re

import nltk
from nltk import WordNetLemmatizer, word_tokenize, SnowballStemmer
from nltk.corpus import wordnet


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


# transform tag forms
def penn_to_wn(tag):
    if is_adjective(tag):
        return wordnet.ADJ
    elif is_noun(tag):
        return wordnet.NOUN
    elif is_adverb(tag):
        return wordnet.ADV
    elif is_verb(tag):
        return wordnet.VERB
    return wordnet.NOUN


def transform(doc: str, stopwords: list, tags: list, wnl: WordNetLemmatizer, stemmer: SnowballStemmer):
    doc = doc.lower()
    # pattern for numbers | words of length=2 | punctuations | words of length=1
    pattern = re.compile(r'[0-9]+|\b[\w]{2,2}\b|[%.,_`!"&?/\\\')({~@;:#}+-]+|\b[\w]{1,1}\b')
    doc = pattern.sub(' ', doc)
    white_space_pattern = re.compile(r'  +')
    doc = white_space_pattern.sub(' ', doc)
    # tokenize document
    doc_tok = word_tokenize(doc)
    # filter out patterns from words
    doc_tok = [x for x in doc_tok if x not in stopwords]
    doc_tok = [pattern.sub('', x) for x in doc_tok]
    # get rid of anything with length=1
    doc_tok = [x.strip() for x in doc_tok if len(x.strip()) > 1]
    # position tagging
    doc_tagged = nltk.pos_tag(doc_tok)
    # selecting nouns and adjectives
    doc_tagged = [(t[0], t[1]) for t in doc_tagged if t[1] in tags]
    # preparing lemmatization
    doc = [(t[0], penn_to_wn(t[1])) for t in doc_tagged]
    # lemmatization
    doc = [wnl.lemmatize(t[0], t[1]) for t in doc]
    # uncomment if you want stemming as well
    doc = [stemmer.stem(x) for x in doc]
    return doc


class LemmaTokenizer(object):
    def __init__(self, stopwords: list, tags: list):
        self.wnl = WordNetLemmatizer()
        # we define (but not use) a stemming method, uncomment the last line in __call__ to get stemming tooo
        self.stemmer = SnowballStemmer('english')
        self.stopwords = stopwords
        self.tags = tags

    def __call__(self, doc):
        return transform(doc, self.stopwords, self.tags, self.wnl, self.stemmer)

    def transform(self, docs):
        return [transform(doc, self.stopwords, self.tags, self.wnl, self.stemmer) for doc in docs]

    def fit(self):
        return self

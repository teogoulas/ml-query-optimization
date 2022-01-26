from typing import Tuple, Any

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from models.lemma_tokenizer import LemmaTokenizer
from models.text_selector import TextSelector


def bag_of_characters(input_texts: list, target_texts: list, input_characters: list, target_characters: list):
    # initialize encoder , decoder input and target data.
    en_in_data = []
    dec_in_data = []
    dec_tr_data = []
    # padding variable with first character as 1 as rest all 0.
    pad_en = [1] + [0] * (len(input_characters) - 1)
    pad_dec = [0] * (len(target_characters))
    pad_dec[2] = 1
    # count vectorizer for one hot encoding as we want to tokenize character so
    # analyzer is true and None the stopwords action.
    cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='word')

    max_input_length = max([len(i) for i in input_texts])
    max_target_length = max([len(i) for i in target_texts])

    for i, (input_t, target_t) in enumerate(zip(input_texts, target_texts)):
        # fit the input characters into the CountVectorizer function
        cv_inp = cv.fit(input_characters)

        # transform the input text from the help of CountVectorizer fit.
        # if character present than put 1 and 0 otherwise.
        en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())
        cv_tar = cv.fit(target_characters)
        dec_in_data.append(cv_tar.transform(list(target_t)).toarray().tolist())
        # decoder target will be one timestep ahead because it will not consider
        # the first character i.e. '\t'.
        dec_tr_data.append(cv_tar.transform(list(target_t)[1:]).toarray().tolist())

        # add padding variable if the length of the input or target text is smaller
        # than their respective maximum input or target length.

        if len(input_t) < max_input_length:
            for _ in range(max_input_length - len(input_t)):
                en_in_data[i].append(pad_en)
        if len(target_t) < max_target_length:
            for _ in range(max_target_length - len(target_t)):
                dec_in_data[i].append(pad_dec)
        if (len(target_t) - 1) < max_target_length:
            for _ in range(max_target_length - len(target_t) + 1):
                dec_tr_data[i].append(pad_dec)

    # convert list to numpy array with data type float32
    en_in_data = np.array(en_in_data, dtype="float32")
    dec_in_data = np.array(dec_in_data, dtype="float32")
    dec_tr_data = np.array(dec_tr_data, dtype="float32")
    return en_in_data, dec_in_data, dec_tr_data


def text_vectorization(data: pd.DataFrame, features: list, ngram_range: Tuple) -> Tuple[CountVectorizer, list]:
    """
        Calculates tf-idf vector
        :param ngram_range: range of examined n-gram
        :param data: text corpus dataframe
        :param features: feature list
        :return: tf-idf vector and tf-idf vectorizer
    """

    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')

    tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR']

    ct_vec = CountVectorizer(ngram_range=ngram_range, analyzer='word', tokenizer=LemmaTokenizer(stopwords, tags))

    pipelined_features = [(feature, Pipeline([('selector', TextSelector(key=feature)), ('vectorizer', ct_vec)])) for
                          feature in features]

    feats = FeatureUnion(pipelined_features)
    feats.fit(data[features])

    pipelined_features = [
        (feature, Pipeline([('selector', TextSelector(key=feature)), ('vectorizer', LemmaTokenizer(stopwords, tags))])) for
        feature in features]

    feats = FeatureUnion(pipelined_features)
    corpus_vec = feats.transform(data[features])

    return ct_vec, corpus_vec

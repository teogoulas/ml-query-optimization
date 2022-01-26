import tempfile
from io import BytesIO

import requests
import zipfile

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def get_glove_vectors(glove_path='data/glove/glove.6B.50d.txt', download_glove=False) -> KeyedVectors:
    if download_glove:
        print('Downloading started')
        # Defining the zip file URL
        url = 'http://nlp.stanford.edu/data/glove.6B.zip'

        # Downloading the file by sending the request to the URL
        req = requests.get(url)
        print('Downloading Completed')

        # extracting the zip file contents
        zipFile = zipfile.ZipFile(BytesIO(req.content))
        zipFile.extractall('../data/glove')

    tmp_file, tmp_filePath = tempfile.mkstemp("test_word2vec.txt")

    _ = glove2word2vec(glove_path, tmp_filePath)

    glove_vectors = KeyedVectors.load_word2vec_format(tmp_filePath)

    return glove_vectors

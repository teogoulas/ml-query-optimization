import os

import pandas as pd

from models.database import Database
from models.embeddings_model import EmbeddingsModel
from models.job_query import JOBQuery
from utils.downloader import get_glove_vectors
from utils.parser import generate_output_text, generate_input_text
from utils.vectorization import text_vectorization


def main():
    dataset = "data/queries"
    cwd = os.getcwd()
    files = os.listdir(os.path.join(cwd, *dataset.split("/")))

    db = Database(collect_db_info=True)
    column_array_index = []
    for table, columns in db.tables_attributes.items():
        for column in columns:
            column_array_index.append(table + "_" + column)

    # initialize all variables
    raw_input_texts = []
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    for file in files:
        f = open(dataset + "/" + file, "r")
        query = f.read().strip()
        raw_input_texts.append(query)
        job_query = JOBQuery(query)
        rows = db.explain_query(query)

        input_text = generate_input_text(job_query.predicates, job_query.rel_lookup)
        input_texts.append(input_text)
        # add '\t' at start and '\n' at end of text.
        target_text = '\t' + generate_output_text(rows, job_query.rel_lookup)[:-1] + '\n'
        target_texts.append(target_text)

    raw_input_vectorizer, raw_input_corpus = text_vectorization(pd.DataFrame(raw_input_texts,
                                                                             columns=['input_queries']),
                                                                ['input_queries'], (1, 3))

    input_vectorizer, input_corpus = text_vectorization(pd.DataFrame(input_texts, columns=['input_queries']),
                                                        ['input_queries'], (1, 1))

    output_vectorizer, output_corpus = text_vectorization(pd.DataFrame(target_texts, columns=['output_queries']),
                                                          ['output_queries'], (1, 3))
    print("number of encoder words : ", len(input_vectorizer.vocabulary_.keys()))
    print("number of decoder words : ", len(output_vectorizer.vocabulary_.keys()))

    glove_vectors = get_glove_vectors()
    input_encoder = EmbeddingsModel()
    input_encoder.build(input_corpus, glove_vectors)
    output_encoder = EmbeddingsModel()
    output_encoder.build(output_corpus, glove_vectors)


if __name__ == "__main__":
    main()

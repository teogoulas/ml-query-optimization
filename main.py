import json
import os
import sys

import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from models.database import Database
from models.decoder import Decoder
from models.embeddings_model import EmbeddingsModel
from models.encoder import Encoder
from models.job_query import JOBQuery
from utils.constants import BATCH_SIZE, EPOCHS, LOG_EVERY
from utils.downloader import get_glove_vectors
from utils.parser import generate_output_text, generate_input_text
from utils.ploting import plot_embeddings
from utils.training import train_step
from utils.vectorization import text_vectorization, pad_tokenizer, get_embedding_matrix


def main(pre_trained: bool):
    if pre_trained:
        input_encoder = EmbeddingsModel()
        input_encoder.model = Word2Vec.load('data/embedding_models/input_encoder.txt')
        output_encoder = EmbeddingsModel()
        output_encoder.model = Word2Vec.load('data/embedding_models/output_encoder.txt')
        input_df = pd.read_csv('data/training/input_data.csv')
        output_df = pd.read_csv('data/training/output_data.csv')
    else:
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
        raw_output_texts = []
        input_texts = []
        target_texts = []

        with open("sql.log", "w", encoding='utf-8') as logf:
            for file in files:
                with open(dataset + "/" + file, "r") as f:
                    queries = f.read().strip()
                    for query in queries.split(";"):
                        if len(query) == 0:
                            continue

                        try:
                            query = query.replace('\n', ' ').strip()
                            raw_input_texts.append(query)
                            job_query = JOBQuery(query)
                            rows = db.explain_query(query)
                            raw_output_texts.append(json.dumps(rows))

                            input_text = generate_input_text(job_query.predicates, job_query.rel_lookup)
                            input_texts.append(input_text)
                            # add '\t' at start and '\n' at end of text.
                            target_text = generate_output_text(rows, job_query.rel_lookup)[:-1]
                            target_texts.append(target_text)
                        except Exception as e:
                            logf.write("Failed to execute query {0}: {1}\n".format(str(query), str(e)))
                            db.conn.close()
                            db.conn = db.connect()
                            if len(input_texts) != len(target_texts):
                                input_texts.pop()
                            if len(raw_input_texts) != len(raw_output_texts):
                                raw_input_texts.pop()
                        finally:
                            pass

        # raw_input_vectorizer, raw_input_corpus = text_vectorization(pd.DataFrame(raw_input_texts,
        #                                                                          columns=['input_queries']),
        #                                                             ['input_queries'], (1, 3))
        raw_input_df = pd.DataFrame(raw_input_texts, columns=['input_queries'])
        raw_output_df = pd.DataFrame(raw_output_texts, columns=['output_queries'])
        raw_input_df.to_csv("data/training/raw_input_data.csv", encoding='utf-8', sep=';')
        raw_output_df.to_csv("data/training/raw_output_data.csv", encoding='utf-8', sep=';')

        input_df = pd.DataFrame(input_texts, columns=['input_queries'])
        output_df = pd.DataFrame(target_texts, columns=['output_queries'])
        input_vectorizer, input_corpus = text_vectorization(input_df, ['input_queries'], (1, 1))
        output_vectorizer, output_corpus = text_vectorization(output_df, ['output_queries'], (1, 3))

        input_df.to_csv("data/training/input_data.csv", encoding='utf-8', sep=',')
        output_df.to_csv("data/training/output_data.csv", encoding='utf-8', sep=',')

        print("number of encoder words : ", len(input_vectorizer.vocabulary_.keys()))
        print("number of decoder words : ", len(output_vectorizer.vocabulary_.keys()))

        glove_vectors = get_glove_vectors()
        input_encoder = EmbeddingsModel()
        input_encoder.build(input_corpus, glove_vectors)
        output_encoder = EmbeddingsModel()
        output_encoder.build(output_corpus, glove_vectors)

        filename = 'data/embedding_models/input_encoder'
        input_encoder.model.save(filename)
        filename = 'data/embedding_models/output_encoder'
        output_encoder.model.save(filename)

    plot_embeddings(input_encoder.model.wv, 27, "Input Encoder Embeddings")
    plot_embeddings(output_encoder.model.wv, 38, "Output Encoder Embeddings")

    X = input_df['input_queries'].values
    y = output_df['output_queries'].values
    # tokenize input
    x_max_length, x_vocab, X_pad = pad_tokenizer(X)
    # tokenize output
    y_max_length, y_vocab, y_pad = pad_tokenizer(y)
    X_train_pad, X_test_pad, y_train_pad, y_test_pad = train_test_split(X_pad, y_pad, test_size=0.2, random_state=42)

    # train model
    input_tensor = tf.convert_to_tensor(X_train_pad)
    output_tensor = tf.convert_to_tensor(y_train_pad)

    buffer_size = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor)).shuffle(buffer_size)
    dataset = dataset.batch(BATCH_SIZE)

    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    encoder = Encoder(len(x_vocab) + 1, get_embedding_matrix(x_vocab, input_encoder.model.wv), x_max_length)
    decoder = Decoder(len(y_vocab) + 1, get_embedding_matrix(y_vocab, output_encoder.model.wv), y_max_length)

    for e in range(1, EPOCHS):

        total_loss = 0.0
        enc_hidden = encoder.init_hidden()

        for idx, (input_tensor, target_tensor) in enumerate(dataset.take(steps_per_epoch)):
            print("idx: {0}, input_tensor shape: {1}, target_tensor shape: {2}".format(idx, input_tensor.shape,
                                                                                       output_tensor.shape))
            batch_loss = train_step(input_tensor, target_tensor, enc_hidden, encoder, decoder)
            total_loss += batch_loss

            if idx % LOG_EVERY == 0:
                print("Epochs: {} batch_loss: {:.4f}".format(e, batch_loss))
                checkpoint(encoder, 'encoder')
                checkpoint(decoder, 'decoder')

        if e % 2 == 0:
            print("Epochs: {}/{} total_loss: {:.4f}".format(
                e, EPOCHS, total_loss / steps_per_epoch))


if __name__ == "__main__":
    args = sys.argv[1:]
    pretrained = len(args) > 0 and args[0] == 'pretrained'
    main(pretrained)

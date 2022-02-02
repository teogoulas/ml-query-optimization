import os
import sys

import tensorflow as tf
import pandas as pd
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
from utils.training import train_step, checkpoint
from utils.vectorization import text_vectorization, pad_tokenizer, get_embedding_matrix, load_embeddings_model


def main(pre_trained: bool):
    if pre_trained:
        input_encoder = EmbeddingsModel()
        input_encoder.model = load_embeddings_model('data/embedding_models/input_encoder.txt')
        output_encoder = EmbeddingsModel()
        output_encoder.model = load_embeddings_model('data/embedding_models/output_encoder.txt')
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

        # raw_input_vectorizer, raw_input_corpus = text_vectorization(pd.DataFrame(raw_input_texts,
        #                                                                          columns=['input_queries']),
        #                                                             ['input_queries'], (1, 3))

        input_df = pd.DataFrame(input_texts, columns=['input_queries'])
        output_df = pd.DataFrame(target_texts, columns=['output_queries'])
        input_vectorizer, input_corpus = text_vectorization(input_df, ['input_queries'], (1, 1))
        output_vectorizer, output_corpus = text_vectorization(output_df, ['output_queries'], (1, 3))

        print("number of encoder words : ", len(input_vectorizer.vocabulary_.keys()))
        print("number of decoder words : ", len(output_vectorizer.vocabulary_.keys()))

        glove_vectors = get_glove_vectors()
        input_encoder = EmbeddingsModel()
        input_encoder.build(input_corpus, glove_vectors)
        output_encoder = EmbeddingsModel()
        output_encoder.build(output_corpus, glove_vectors)

    plot_embeddings(input_encoder.model.mv, 27, "Input Encoder Embeddings")
    plot_embeddings(output_encoder, 38, "Output Encoder Embeddings")

    filename = 'data/embedding_models/input_encoder.txt'
    input_encoder.model.wv.save_word2vec_format(filename, binary=False)
    filename = 'data/embedding_models/output_encoder.txt'
    output_encoder.model.wv.save_word2vec_format(filename, binary=False)

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
    hidden = tf.zeros((16, 256))
    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    encoder = Encoder(len(x_vocab) + 1, get_embedding_matrix(x_vocab, input_encoder.model.wv), x_max_length)
    decoder = Decoder(len(y_vocab) + 1, get_embedding_matrix(y_vocab, output_encoder.model.wv), y_max_length)

    for e in range(1, EPOCHS):

        total_loss = 0.0
        enc_hidden = encoder.init_hidden()

        for idx, (input_tensor, target_tensor) in enumerate(dataset.take(steps_per_epoch)):
            print("idx: {0}, input_tensor shape: {1}, target_tensor shape: {2}".format(idx, input_tensor.shape,
                                                                                       output_tensor.shape))
            batch_loss = train_step(input_tensor, target_tensor, hidden, encoder, decoder)
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

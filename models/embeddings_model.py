from gensim.models import Word2Vec, KeyedVectors


class EmbeddingsModel(object):
    def __init__(self, vector_size=300, window=5, min_count=5, workers=-1,
                 epochs=20, sg=0, seed=17, cbow_mean=1, alpha=0.025, sample=0.001):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self.sg = sg
        self.cbow_mean = cbow_mean
        self.alpha = alpha
        self.sample = sample
        self.model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, null_word=0,
                              workers=workers, seed=seed, sg=sg, cbow_mean=cbow_mean, alpha=alpha, sample=sample)

    def build(self, tokens: list, glove_vectors: KeyedVectors):
        self.model.build_vocab(tokens)
        total_examples = self.model.corpus_count
        self.model.build_vocab([list(glove_vectors.index_to_key)], update=True)
        self.model.train(tokens, total_examples=total_examples, epochs=self.epochs)

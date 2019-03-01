from text_vectorian import Token, Vectorizer, Tokenizer, TextVectorian
from gensim.models.word2vec import Word2Vec
import numpy as np
import yaml
import os
import text_vectorian.utils as utils
from logging import getLogger

logger = getLogger(__name__)
DEFAULT_TOKEN = 'の'

config = utils.load_config()

class Char2VecVectorizer(Vectorizer):
    def __init__(self, model_filename):
        self._model = Word2Vec.load(model_filename)
        self._index2token = self._model.wv.index2word
        self._token2index = { c: i for i, c in enumerate(self._index2token) }
    def _vectorize(self, token: str):
        if token == ' ':
            normalized_token = DEFAULT_TOKEN
        else:
            normalized_token = token.lower()
        if normalized_token not in self.token2index:
            logger.warning(f'{token} was not in vecabs, so use default token({DEFAULT_TOKEN}).')
            normalized_token = DEFAULT_TOKEN
        index = self.token2index[normalized_token]
        vector = self.model[normalized_token]

        return (index, vector)
    def _get_keras_layer(self, trainable):
        layer = self.model.wv.get_keras_embedding(train_embeddings=trainable)

        return layer
    @property
    def model(self):
        return self._model
    @property
    def token2index(self):
        return self._token2index
    @property
    def index2token(self):
        return self._index2token

class CharacterTokenizer(Tokenizer):
    def __init__(self):
        pass
    def _tokenize(self, text: str):
        return list(text)

class Char2VecVectorian(TextVectorian):
    def __init__(self):
        self._vectorizer_filename = utils.load_model('char2vec', 'vectorizer', config)[0]
        self._tokenizer = CharacterTokenizer()
        self._vectorizer = Char2VecVectorizer(self._vectorizer_filename)
    @property
    def tokenizer(self):
        return self._tokenizer
    @property
    def vectorizer(self):
        return self._vectorizer
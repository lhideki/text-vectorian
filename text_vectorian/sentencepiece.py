from text_vectorian import Token, Vectorizer, Tokenizer, TextVectorian
import sentencepiece as spm
from gensim.models.word2vec import Word2Vec
import numpy as np
import yaml
import os
import text_vectorian.utils as utils
from logging import getLogger

logger = getLogger(__name__)
DEFAULT_TOKEN = '▁'

config = utils.load_config()

class Word2VecVectorizer(Vectorizer):
    def __init__(self, model_filename):
        self._model = Word2Vec.load(model_filename)
        self._index2token = self._model.wv.index2word
        self._token2index = { c: i for i, c in enumerate(self._index2token) }
    def _vectorize(self, token: str):
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
class SentencePieceTokenizer(Tokenizer):
    def __init__(self, tokenizer_filename: str):
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_filename)
        self._tokenizer = sp
    def _tokenize(self, text: str):
        tokens = self._tokenizer.EncodeAsPieces(text)

        return tokens
class SentencePieceVectorian(TextVectorian):
    def __init__(self, tokenizer_filename = None, vectorizer_filename = None):
        if tokenizer_filename:
            self._tokenizer_filename = tokenizer_filename
        else:
            self._tokenizer_filename = utils.load_model('sentencepiece', 'tokenizer', config)[0]
        if vectorizer_filename:
            self._vectorizer_filename = vectorizer_filename
        else:
            self._vectorizer_filename = utils.load_model('sentencepiece', 'vectorizer', config)[0]
        self._tokenizer = SentencePieceTokenizer(self._tokenizer_filename)
        self._vectorizer = Word2VecVectorizer(self._vectorizer_filename)
    @property
    def tokenizer(self):
        return self._tokenizer
    @property
    def vectorizer(self):
        return self._vectorizer
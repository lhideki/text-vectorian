from abc import ABC, abstractmethod
import numpy as np

class Vectorizer(ABC):
    @abstractmethod
    def _vectorize(self, token: str):
        pass
    @abstractmethod
    def _get_keras_layer(self, trainable):
        pass
class Token:
    def __init__(self, text: str, vectorizer: Vectorizer):
        self._text = text
        (self._index, self._vector) = vectorizer._vectorize(text)
    @property
    def text(self):
        return self._text
    @property
    def vector(self):
        return self._vector
    @property
    def index(self):
        return self._index
class Tokenizer(ABC):
    @abstractmethod
    def _tokenize(self, text: str):
        pass
    def _create_tokens(self, text: str, vectorizer: Vectorizer):
        tokens = self._tokenize(text)

        return [Token(token, vectorizer) for token in tokens]
class TextVectorian(ABC):
    @property
    @abstractmethod
    def tokenizer(self):
        pass
    @property
    @abstractmethod
    def vectorizer(self):
        pass
    @property
    def max_tokens_len(self):
        return self._max_tokens_len
    def reset(self):
        self._max_tokens_len = 0
    def fit(self, text: str):
        self._tokens = self.tokenizer._create_tokens(text, self.vectorizer)
        if hasattr(self, '_max_tokens_len'):
            self._max_tokens_len = max(len(self._tokens), self._max_tokens_len)
        else:
            self._max_tokens_len = len(self._tokens)
        self._vectors = []
        self._indices = []

        for token in self._tokens:
            self._vectors.append(token.vector)
            self._indices.append(token.index)

        return self
    @property
    def tokens(self):
        return self._tokens
    @property
    def vectors(self):
        return np.array(self._vectors)
    @property
    def indices(self):
        return np.array(self._indices)
    def get_keras_layer(self, trainable=False):
        return self.vectorizer._get_keras_layer(trainable)
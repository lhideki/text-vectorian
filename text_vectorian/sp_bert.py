from text_vectorian import Token, Vectorizer, Tokenizer, TextVectorian, SentencePieceTokenizer
import keras_bert
import numpy as np
import json
from logging import getLogger

logger = getLogger(__name__)

class SpBertVectorizer(Vectorizer):
    def __init__(self, model_filename, config_filename):
        self._model = keras_bert.load_trained_model_from_checkpoint(config_filename, model_filename)
    def _vectorize(self, token: str):
        raise NotImplementedError()
    def _get_keras_layer(self, trainable):
        raise NotImplementedError()
    @property
    def model(self):
        return self._model

class SpBertVectorian(TextVectorian):
    def __init__(self, tokenizer_filename, vectorizer_filename, vectorizer_config_filename):
        self._tokenizer_filename = tokenizer_filename
        self._vectorizer_filename = vectorizer_filename
        self._vectorizer_config_filename = vectorizer_config_filename
        with open(self._vectorizer_config_filename) as f:
            self._config = json.load(f)
        self._tokenizer = SentencePieceTokenizer(self._tokenizer_filename)
        self._vectorizer = SpBertVectorizer(self._vectorizer_filename, self._vectorizer_config_filename)
    def fit(self, text: str):
        sp = self.tokenizer._tokenizer
        tokens = []
        tokens.append('[CLS]')
        tokens.extend(sp.encode_as_pieces(text))
        tokens.append('[SEP]')
        self._tokens = tokens
        common_seg_input = np.zeros((1, self.max_tokens_len), dtype = np.float32)
        indices = np.zeros((1, self.max_tokens_len), dtype=np.float32)

        for i, token in enumerate(tokens):
            try:
                indices[0, i] = sp.piece_to_id(token)
            except:
                logger.warning(f'{token} was not in vecabs, so use default token(<unk>).')
                indices[0, i] = sp.piece_to_id('<unk>')
        self._indices = indices
        self._vectors =  self.vectorizer.model.predict([indices, common_seg_input])[0]

        return self
    @property
    def max_tokens_len(self):
        return self._config['max_seq_length']
    @property
    def tokenizer(self):
        return self._tokenizer
    @property
    def vectorizer(self):
        return self._vectorizer
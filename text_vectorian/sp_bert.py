from text_vectorian import Token, Vectorizer, Tokenizer, TextVectorian, SentencePieceTokenizer
from keras_bert import load_trained_model_from_checkpoint, calc_train_steps, AdamWarmup
import numpy as np
import json
from logging import getLogger
import os

logger = getLogger(__name__)


class SpBertVectorizer(Vectorizer):
    def __init__(self, model_filename, config_filename):
        self._model_filename = model_filename
        self._config_filename = config_filename
        self._model = load_trained_model_from_checkpoint(
            config_filename, model_filename)

    def _vectorize(self, token: str):
        raise NotImplementedError()

    def _get_keras_layer(self, trainable):
        self._model = load_trained_model_from_checkpoint(
            self._config_filename, self._model_filename, training=trainable)

        return {
            'inputs': [self._model.input[0], self._model.input[1]],
            'last': self._model.get_layer(name='NSP-Dense').output
        }

    @property
    def model(self):
        return self._model


class SpBertVectorian(TextVectorian):
    def __init__(self, tokenizer_filename, vectorizer_filename, config_filename=f'{os.path.dirname(os.path.abspath(__file__))}/default_bert_config.json'):
        self._tokenizer_filename = tokenizer_filename
        self._vectorizer_filename = vectorizer_filename
        self._config_filename = config_filename
        with open(self._config_filename) as f:
            self._config = json.load(f)
        self._tokenizer = SentencePieceTokenizer(self._tokenizer_filename)
        self._vectorizer = SpBertVectorizer(
            self._vectorizer_filename, self._config_filename)

    def fit(self, text: str, suppress_vectors: bool = False):
        self._suppress_vectors = suppress_vectors
        sp = self.tokenizer._tokenizer
        input_tokens = []
        input_tokens.append('[CLS]')

        capable_text_tokens_len = self.max_tokens_len - 2
        text_tokens = sp.encode_as_pieces(text)[:]
        if len(text_tokens) > capable_text_tokens_len:
            logger.warning(
                f'Text tokens len is too long than config tokens len({len(text_tokens)}/{capable_text_tokens_len}).')
            text_tokens = text_tokens[:capable_text_tokens_len]

        input_tokens.extend(text_tokens)
        input_tokens.append('[SEP]')
        self._tokens = input_tokens
        common_seg_input = np.zeros((1, self.max_tokens_len), dtype=np.float32)
        indices = np.zeros((self.max_tokens_len), dtype=np.float32)

        for i, token in enumerate(self._tokens):
            try:
                indices[i] = sp.piece_to_id(token)
            except:
                logger.warning(
                    f'{token} was not in vocabs, so use default token(<unk>).')
                indices[i] = sp.piece_to_id('<unk>')
        self._indices = indices
        if not self._suppress_vectors:
            self._vectors = self.vectorizer.model.predict(
                [[indices], common_seg_input])[0]
        if hasattr(self, '_samples_len'):
            self._samples_len += 1
        else:
            self._samples_len = 1

        return self

    @property
    def max_tokens_len(self):
        return self._config['max_seq_length']

    @property
    def vectors(self):
        if self._suppress_vectors:
            raise NotImplementedError(
                "You specified suppress_vectors, so you cannot get vectors.")
        return self._vectors

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vectorizer(self):
        return self._vectorizer

    def get_optimizer(self, samples_len, batch_size, epochs, lr=1e-4):
        decay_steps, warmup_steps = calc_train_steps(
            samples_len,
            batch_size=batch_size,
            epochs=epochs,
        )
        optimizer = AdamWarmup(
            decay_steps=decay_steps, warmup_steps=warmup_steps, lr=lr)

        return optimizer

    def get_segments(self):
        segments = np.zeros((self.samples_len, self.max_tokens_len))

        return segments

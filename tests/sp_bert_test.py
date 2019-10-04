from text_vectorian import SpBertVectorian
import unittest
from logging import getLogger
import keras
from pprint import pprint
import numpy as np

logger = getLogger(__name__)
logger.setLevel('INFO')


class SpBertVectorianTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 以下より各ファイルをダウンロードして、../bert-japanese/modelに配置してください。
        # https://yoheikikuta.github.io/bert-japanese/
        tokenizer_filename = '../bert-japanese/model/wiki-ja.model'
        vectorizer_filename = '../bert-japanese/model/model.ckpt-1400000'
        cls.vectorian = SpBertVectorian(
            tokenizer_filename=tokenizer_filename,
            vectorizer_filename=vectorizer_filename
        )

    def test_vectorを取得できる(self):
        test_text = 'これはテストです。'
        vectors = self.vectorian.fit(test_text).vectors

        expected = (128, 768)
        fact = vectors.shape
        self.assertTupleEqual(fact, expected)

    def test_configで指定したトークン長より長いトークンを指定すると警告ログが出力される(self):
        test_text = 'これはテストです。' * 34
        with self.assertLogs(getLogger('text_vectorian.sp_bert'), 'WARN'):
            vectors = self.vectorian.fit(test_text).vectors

    def test_indexを取得できる(self):
        test_text = 'これはテストです。'
        fact = self.vectorian.fit(test_text).indices
        expected = [4, 444, 2666, 2767, 8, 5]

        self.assertListEqual(fact[:6].tolist(), expected)

    def test_samples_lenを取得できる(self):
        test_text = 'これはテストです。'
        self.vectorian.reset()
        samples_len = self.vectorian.fit(test_text).samples_len
        self.assertEquals(samples_len, 1)
        samples_len = self.vectorian.fit(test_text).samples_len
        self.assertEquals(samples_len, 2)

    def test_segmentsを取得できる(self):
        test_text = 'これはテストです。'
        self.vectorian.reset()
        self.vectorian.fit(test_text).samples_len
        self.vectorian.fit(test_text).samples_len
        fact = self.vectorian.get_segments()
        expected = np.zeros((2, 128))

        self.assertListEqual(fact.tolist(), expected.tolist())

    def test_トレーニング用のkerasレイヤーが取得できる(self):
        layers = self.vectorian.get_keras_layer(trainable=True)
        optimizer = self.vectorian.get_optimizer(10, 1, 1)

        output_tensor = keras.layers.Dense(10)(layers['last'])
        model = keras.Model(layers['inputs'], output_tensor)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        # model.summary()
        config = model.get_config()
        self.assertListEqual(config['output_layers'], [['dense_1', 0, 0]])

    def test_suppress_vectors指定時にvectorsを取得すると例外が発生する(self):
        test_text = 'これはテストです。'
        with self.assertRaises(NotImplementedError):
            self.vectorian.fit(test_text, suppress_vectors=True).vectors


if __name__ == '__main__':
    unittest.main(exit=False)

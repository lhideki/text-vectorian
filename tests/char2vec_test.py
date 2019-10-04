from text_vectorian import Char2VecVectorian
import unittest
from logging import getLogger
import keras

logger = getLogger(__name__)
logger.setLevel('INFO')


class Char2VecVectorianTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectorian = Char2VecVectorian()

    def test_Tokenに分割できる(self):
        test_text = 'これはテストです。'
        self.vectorian.fit(test_text)

        expected = [
            'こ', 'れ', 'は', 'テ', 'ス', 'ト', 'で', 'す', '。'
        ]
        fact = [token.text for token in self.vectorian.tokens]
        self.assertListEqual(fact, expected)

    def test_indexを取得できる(self):
        test_text = 'これはテストです。'
        self.vectorian.fit(test_text)

        expected = [37, 17, 6, 87, 22, 30, 11, 39, 3]
        fact = self.vectorian.indices
        self.assertListEqual(fact.tolist(), expected)

    def test_vectorを取得できる(self):
        test_text = 'テスト'
        self.vectorian.fit(test_text)

        expected = (3, 30)
        fact = self.vectorian.vectors.shape
        self.assertTupleEqual(fact, expected)

    def test_samples_lenを取得できる(self):
        test_text = 'テスト'
        self.vectorian.reset()

        self.vectorian.fit(test_text)
        self.assertEquals(self.vectorian.samples_len, 1)
        self.vectorian.fit(test_text)
        self.assertEquals(self.vectorian.samples_len, 2)

    def test_OutOfVocab時にログが出力される(self):
        test_text = '⌛️'
        with self.assertLogs(level='WARN') as cm:
            self.vectorian.fit(test_text)
            print(cm.output)

    def test_複数回fitした場合にTokenの最大数を取得できる(self):
        test_text1 = 'これはテスト'
        test_text2 = 'これはテストです。'

        self.vectorian.reset()
        self.assertEquals(self.vectorian.max_tokens_len, 0)
        self.vectorian.fit(test_text1)
        self.assertEquals(self.vectorian.max_tokens_len, 6)
        self.vectorian.fit(test_text2)
        self.assertEquals(self.vectorian.max_tokens_len, 9)
        self.vectorian.reset()
        self.assertEquals(self.vectorian.max_tokens_len, 0)

    def test_kerasのlayerが取得できる(self):
        layer = self.vectorian.get_keras_layer()
        self.assertTrue(isinstance(layer, keras.layers.embeddings.Embedding))


if __name__ == '__main__':
    unittest.main(exit=False)

from text_vectorian import SpBertVectorian
import unittest
from logging import getLogger
import keras

logger = getLogger(__name__)
logger.setLevel('INFO')

class SpBertVectorianTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 以下より各ファイルをダウンロードして、../bert-japanese/modelに配置してください。
        # https://yoheikikuta.github.io/bert-japanese/
        tokenizer_filename = '../bert-japanese/model/wiki-ja.model'
        vectorizer_filename = '../bert-japanese/model/model.ckpt-1400000'
        # configはテスト用の設定を同梱しています。
        vectorizer_config_filename = 'tests/bert_japanese_config.json'
        cls.vectorian = SpBertVectorian(
            tokenizer_filename=tokenizer_filename,
            vectorizer_filename=vectorizer_filename,
            vectorizer_config_filename=vectorizer_config_filename
        )
    def test_vectorを取得できる(self):
        test_text = 'これはテストです。'
        vectors = self.vectorian.fit(test_text).vectors

        expected = (64, 768)
        fact = vectors.shape
        self.assertTupleEqual(fact, expected)

if __name__ == '__main__':
    unittest.main(exit=False)
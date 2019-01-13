from text_vectorian import SentencePieceVectorian

vectorian = SentencePieceVectorian()
text = 'これはテストです。'
indices = vectorian.fit(text).indices

print(indices)

from keras import Input, Model
from keras.layers import Dense, LSTM

input_tensor = Input((vectorian.max_tokens_len,))
common_input = vectorian.get_keras_layer(trainable=True)(input_tensor)
l1 = LSTM(32)(common_input)
output_tensor = Dense(3)(l1)

model = Model(input_tensor, output_tensor)
model.summary()
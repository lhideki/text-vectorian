from text_vectorian import SentencePieceVectorian

vectorian = SentencePieceVectorian()
text = 'これはテストです。'
vectors = vectorian.fit(text).vectors

print(vectors)
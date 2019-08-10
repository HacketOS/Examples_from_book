import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

top_words = 5000
max_review_length = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = top_words)
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embeddings_vectors_length = 32
model = Sequential()
model.add(Embedding(top_words, embeddings_vectors_length,input_length = max_review_length ))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation='relu'))
model.add(LSTM(100))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
model.fit(X_train, y_train, verbose = 0)
scores = model.evaluate(X_test, y_test)
print('Accuracy: %.5f' %scores[1])
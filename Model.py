from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D,Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from numpy import vstack, row_stack, asarray
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from pandas import read_csv
from pymystem3 import Mystem
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from collections import Counter










def create_ngram_set(input_list, ngram_value=2):

    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):

    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def load_twitter_msgs():
    names = ['id', 'tdate', 'tname', 'ttext', 'ttype', 'trep',
             'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount', 'basename']
    data_negative = read_csv('./twitter/negative.csv', delimiter=';', names=names)
    data_positive = read_csv('./twitter/positive.csv', delimiter=';', names=names)
    data = row_stack((data_negative[['ttext']], data_positive[['ttext']])).ravel()
    values = row_stack((data_negative[['ttype']], data_positive[['ttype']])).ravel()
    values = [v == 1 for v in values]
    data = asarray(data)
    return data, values
data, values = load_twitter_msgs()

def preprocess(sentence):
    mystem = Mystem()
    sentence=''.join(mystem.lemmatize(sentence))
    sentence=sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    sentence=re.sub(r'@\w+',' ', sentence)
    sentence = re.sub(r'(?:https?:\/\/)?(?:[\w\.]+)\.(?:[a-z]{2,6}\.?)(?:\/[\w\.]*)*\/?', ' ', sentence)
    sentence = re.sub(r'r\w+', ' ', sentence)
    tokens = tokenizer.tokenize(sentence)
    filt_word = [word for word in tokens if word not in stopwords.words('russian')]
    return " ".join(filt_word)





np.random.seed(42)
# set parameters:
ngram_range = 3
max_features = 20000
maxlen = 800
batch_size = 64
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2


def preprocess(sentence):
    mystem = Mystem()
    sentence=''.join(mystem.lemmatize(sentence))
    sentence=sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    sentence=re.sub(r'@\w+',' ', sentence)
    sentence = re.sub(r'(?:https?:\/\/)?(?:[\w\.]+)\.(?:[a-z]{2,6}\.?)(?:\/[\w\.]*)*\/?', ' ', sentence)
    sentence = re.sub(r'r\w+', ' ', sentence)
    tokens = tokenizer.tokenize(sentence)
    filt_word = [word for word in tokens if word not in stopwords.words('russian')]
    return " ".join(filt_word)


def get_sequences(data, values):
    return train_test_split(data, values, train_size=0.9, test_size=0.1)


if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    x_train, x_test, y_train, y_test = get_sequences(data, values)
    y_train = asarray(y_train, dtype=bool)
    y_test = asarray(y_test, dtype=bool)



    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))



print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)



print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='linear',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('linear'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])





history =model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test), verbose=2)


# # Проверяем качество обучения на тестовых данных
scores = model.evaluate(x_test, y_test,
                        batch_size=64)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))


print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

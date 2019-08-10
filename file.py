import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
import os

#preparing
filename = 'data.txt'
START_CHAR = '\b'
END_CHAR = '\t'
PADDING_CHAR = '\a'
chars = set([START_CHAR, '\n', END_CHAR])
with open(filename, 'r', encoding = 'cp1251') as f:
    for line in f:
        chars.update( list(line.strip().lower()) )
char_indeces = {c: i for i,c in enumerate(sorted(list(chars))) }
char_indeces[PADDING_CHAR] = 0
indices_to_chars = {i: c for c,i in char_indeces.items()}
num_chars = len(chars)
# ohe-encoding for symbols
def get_one(i, sz):
    res = np.zeros(sz)
    res[i] = 1
    return res

char_vectors =  {
                c: (np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars))
                for c,v in char_indeces.items()
                }
# parse sentences
sentences_end_markers = set('?.!')
sentences = []
current_sentence = ''
with open(filename, 'r', encoding = 'cp1251') as f:
    for line in f:
        s = line.strip().lower()
        if len(s) > 0:
            current_sentence += s + '\n'
        if len(s) == 0 or s[-1] in sentences_end_markers:
            current_sentence = current_sentence.strip()
            if len(current_sentence) > 10:
                sentences.append(current_sentence)
            current_sentence = ''
# make X and y vectors
def get_matrices(sentences):
    max_sentence_length =np.max([len(x) for x in sentences])
    X = np.zeros((len(sentences), max_sentence_length, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), max_sentence_length, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        char_seq = (START_CHAR + sentence + END_CHAR).ljust(max_sentence_length + 1, PADDING_CHAR)
        for t in range(max_sentence_length):
            X[i,t,:] = char_vectors[char_seq[t]]
            y[i,t,:] = char_vectors[char_seq [t + 1]]
    return X, y 
# making model
model = Sequential()
model.add(LSTM(output_dim = 128, activation='tanh', return_sequences = True, input_dim = num_chars))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(output_dim = num_chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer = Adam(clipnorm = 1.),metrics=['accuracy'])

test_indices = np.random.choice(range(len(sentences)), int(len(sentences)*0.05))
sentences_train = [sentences[x] for x in set(range(len(sentences))) - set(test_indices)]
sentences_test = [sentences[x] for x in test_indices]
sentences_train = sorted(sentences_train, key = lambda x:len(x))
X_test, y_test = get_matrices(sentences_test)
batch_size = 16

def generate_batch():
    while True:
        for i in range( int(len(sentences_train) / batch_size) ):
            sentences_batch = sentences_train[ i*batch_size: (i+1)*batch_size]
            yield get_matrices(sentences_batch)

output_fname = 'idk.txt'

class CharSampler(Callback):
    def __init__(self, char_vectors, model):
        self.char_vectors = char_vectors
        self.model = model

    def on_train_begin(self,logs = {}):
        self.epoch = 0
        if os.path.isfile(output_fname):
            os.remove(output_fname)
        
    def sample(self, preds, temperature = 1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def sample_one(self, T):
        result = START_CHAR
        while len(result) < 500:
            Xsamples = np.zeros((1, len(result), num_chars))
            for t,c in enumerate(list(result)):
                Xsamples[0,t,:] = self.char_vectors[c]
            ysampled = self.model.predict(Xsamples, batch_size = 1)[0,:]
            yv = ysampled[len(result) - 1, :]
            selected_char = indices_to_chars[self.sample(yv, T)]
            if selected_char == END_CHAR:
                break
            result += selected_char
            return selected_char

    def on_epoch_end(self, batch, logs = {}):
        self.epoch += 1
        if self.epoch % 1 == 0:
            print("\n Epoch %d text sampling:" % self.epoch)
            with open(output_fname, 'a') as f:
                f.write("\n === Epoch %d ===" % self.epoch)
                for T in [0.3, 0.5, 0.7, 0.9, 1.1]:
                    print("\tsampling, T = %.1f" % T)
                    for _ in range(5):
                        self.model.reset_states()
                        res = self.sample_one(T)
                        f.write('\nT = %.1f\n%s\n' % (T, res[1:]))


cb_logger = CSVLogger('sin_l/' + filename + '.log')
cb_sampler = CharSampler(char_vectors, model)

model.fit_generator(generate_batch(),int(len(sentences_train)/batch_size) *batch_size,epochs = 10, verbose = True,validation_data = (X_test, y_test),callbacks = [cb_sampler, cb_sampler])

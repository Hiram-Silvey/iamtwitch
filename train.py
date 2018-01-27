# Larger LSTM Network to Generate Text for Alice in Wonderland
import sys, os, random, numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

SEQ_LEN = 100
MAX_BYTES = 200000

# load twitch chat as individual bytes
data_dir = "data/"
files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
raw = []
curr_len = 0
for f in files:
    with open(data_dir+f, 'r') as op_f:
        for line in op_f:
            curr_bytes = line.split('\t')[2].encode()
            raw.append(curr_bytes)
            curr_len += len(curr_bytes)
            if curr_len >= MAX_BYTES:
                break
    if curr_len >= MAX_BYTES:
        break
    # append a null byte to signal change in data
    raw.append(b'\x00')
raw = b''.join(raw)
# create mapping of unique chars to integers
chars = set(list(raw))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw)
n_vocab = len(chars)
print("Total characters: ", n_chars)
print("Total vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
n_seqs = n_chars-SEQ_LEN
for i in random.sample(range(0, n_seqs), n_seqs):
    seq_in = raw[i:i+SEQ_LEN]
    if b'\x00' in seq_in:
        continue
    seq_out = raw[i+SEQ_LEN]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, SEQ_LEN, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)

# Load Larger LSTM network and generate text
import sys, os
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
data_dir = "data/"
files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
raw_text = ""
for f in files:
    with open(data_dir+f, 'r') as op_f:
        for line in op_f:
            raw_text += line.split('\t')[2]
    if len(raw_text) > 150000:
        break
raw_text = raw_text.encode()
charbytes = {}
for byte in raw_text:
    charbytes[byte] = charbytes.get(byte, 0) + 1
# create mapping of unique chars to integers
chars = sorted(charbytes, key=charbytes.get, reverse=True)
char_to_int = dict((c, i) for i, c in enumerate(chars[:256]))
int_to_char = dict((i, c) for i, c in enumerate(chars[:256]))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
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
# load the network weights
filename = "weights-improvement-49-1.5728-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", bytes([int_to_char[value] for value in pattern]), "\"")
gen = []
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	gen.append(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
gen = bytes(gen)
print(gen)
print("\nDone.")

# Loading Dependencies and its Utilities

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# Load the dataset : IMDB
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Preprocessind the data using sequence padding technique : Zeros PADD
# Choosing Maximum length 0
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

# Transform the class Labels into binary format
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Building a Deep learning network

# Initilialize the input layer
net = tflearn.input_data([None, 100])

# Embedding the Recurrent Neural Network (RNN) layer
net = tflearn.embedding(net, input_dim=10000, output_dim=128)

# Use LSTM with a  Regularizaion Dropout of 80%

net = tflearn.lstm(net, 128, dropout=0.8)
# Use softmax Activation function

net = tflearn.fully_connected(net, 2, activation='softmax')
# Use 'adam' optimizer to check gradient descent with a slow learning rate
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training the Deep Learning network 
model = tflearn.DNN(net, tensorboard_verbose=0)

# Fit the Model and Validate using testing data
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)

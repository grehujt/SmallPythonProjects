import os
import argparse
import random

import numpy as np
from keras.models import Model, load_model
from keras.callbacks import Callback, ModelCheckpoint,TensorBoard
from keras.layers import Input, CuDNNGRU, TimeDistributed, Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import multi_gpu_model


CHUNK_SIZE = 256

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, default='input.txt', help='path to input txt file')
ap.add_argument('-l', '--layers', type=int, default=3, help='num of rnn layers')
ap.add_argument('-b', '--batchSize', type=int, default=256, help='batch size')
ap.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
ap.add_argument('-t', '--tensorboard', type=bool, default=False, help='need tensorboard')
ap.add_argument('-m', '--model', type=str, default=None, help='init model')
ap.add_argument('-o', '--output', type=str, default='./models', help='path to output folder')
args = vars(ap.parse_args())
print(args)


class EpochModelCheckPoint(Callback):
    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 9 == 0:
            self.model_to_save.save(
                os.path.join(args['output'], 'model.%03d-{val_loss:.2f}.h5' % epoch))


def char_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.2):
    input = Input(shape=(None, num_chars), name='input')
    prev = input
    for i in range(num_layers):
        prev = CuDNNGRU(num_nodes, return_sequences=True)(prev)
        prev = Dropout(dropout)(prev)
    dense = TimeDistributed(Dense(num_chars, name='dense', activation='softmax'))(prev)
    model = Model(inputs=[input], outputs=[dense])
    # optimizer = Adam(lr=0.002)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def data_generator(all_txt, num_chars, batch_size):
    X = np.zeros((batch_size, CHUNK_SIZE, num_chars))
    y = np.zeros((batch_size, CHUNK_SIZE, num_chars))
    while True:
        for row in range(batch_size):
            idx = random.randrange(len(all_txt)-CHUNK_SIZE-1)
            chunk = np.zeros((CHUNK_SIZE+1, num_chars))
            for i in range(CHUNK_SIZE+1):
                chunk[i, char_to_idx[all_txt[idx+i]]] = 1
            X[row, :, :] = chunk[:CHUNK_SIZE]
            y[row, :, :] = chunk[1:]
            yield X, y


all_txt = open(args['input']).read()
chars = list(sorted(set(all_txt)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
num_chars = len(chars)

checkpoint1 = ModelCheckpoint(
    filepath=os.path.join(args['output'], 'model.{epoch:03d}.h5'),
    period=10)
checkpoint2 = ModelCheckpoint(
    filepath=os.path.join(args['output'], 'model.best.h5'),
    save_best_only=True)
callbacks = [checkpoint1, checkpoint2]
if args['tensorboard']:
    callbacks.append(TensorBoard(histogram_freq=1))

if args['model'] is not None:
    print('loading model:', args['model'])
    model = load_model(args['model'])
else:
    print('init model')
    model = char_rnn_model(num_chars, args['layers'])
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002), metrics=['accuracy'])
model.fit_generator(
    data_generator(all_txt, num_chars, batch_size=args['batchSize']),
    epochs=args['epochs'],
    steps_per_epoch=2*len(all_txt)/(256*CHUNK_SIZE),
    callbacks=callbacks,
    verbose=2
)

# checkpoint = EpochModelCheckPoint(model)
# parallel_model = multi_gpu_model(model, gpus=2)
# parallel_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002), metrics=['accuracy'])
# parallel_model.fit_generator(
#     data_generator(all_txt, num_chars, batch_size=args['batchSize']),
#     epochs=args['epochs'],
#     steps_per_epoch=2*len(all_txt)/(256*CHUNK_SIZE),
#     callbacks=[checkpoint],
#     verbose=2
# )

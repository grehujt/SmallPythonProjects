import argparse
import random
import time

import numpy as np
from keras.models import load_model

CHUNK_SIZE = 256


def generate(model, start_index, amount):
    if start_index is None:
        start_index = random.randint(0, len(all_txt)-CHUNK_SIZE-1)
    fragment = all_txt[start_index : start_index+CHUNK_SIZE]
    result = list(fragment)
    for i in range(amount):
        x = np.zeros((1, CHUNK_SIZE, num_chars))
        for t, char in enumerate(fragment):
            x[0, t, char_to_idx[char]] = 1.
        preds = np.asarray(model.predict(x, verbose=0)[0, -1, :])
        next_index = np.argmax(preds)
        next_char = chars[next_index]
        result.append(next_char)
        fragment = fragment[1:] + next_char
    return start_index, ''.join(result)


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, default='./models/model.100.h5', help='path to model h5 file')
ap.add_argument('-n', '--num', type=int, default=100, help='num of output length')
ap.add_argument('-s', '--start_index', type=int, default=None, help='start index')
ap.add_argument('-i', '--input', type=str, default='input.txt', help='path to input txt file')

args = vars(ap.parse_args())
print(args)

all_txt = open(args['input']).read()
chars = list(sorted(set(all_txt)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
num_chars = len(chars)

print('loading model', args['model'])
model = load_model(args['model'])
print('generating txt')
t = time.time()
start_index, txt = generate(model, args['start_index'], args['num'])

print('done, took {}s'.format(time.time()-t))
with open('gen.txt', 'wb') as fout:
    fout.write(txt.encode('utf8'))
with open('ori.txt', 'wb') as fout:
    fout.write(all_txt[start_index:start_index+args['num']+CHUNK_SIZE].encode('utf8'))

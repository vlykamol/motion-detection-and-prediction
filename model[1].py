import keras
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
import keras.backend as K

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import json

with open('dataset.txt',"r") as fp:
	data = json.load(fp)

	input = np.array(data["mfcc"])
	target = np.array(data["x,y"])

# print(keras.__version__)
# print(input.shape[1], input.shape[2])
no_frames = input.shape[1]
no_features = input.shape[2]
# print(target.shape[1])

model = keras.Sequential()
# model.add(Flatten(input_shape=(input.shape[1], input.shape[2])))
model.add(LSTM(512, activation='relu', input_shape=(no_frames, no_features)))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')
print(K.eval(model.optimizer.lr))

history = model.fit(input, target, epochs=1000, validation_split=0.2, verbose=1)

test_input = np.array([[
                -901.349853515625,
                0.6423366069793701,
                7.134531021118164,
                10.805811882019043,
                8.881223678588867,
                5.810683250427246,
                -4.136382102966309,
                -2.722184181213379,
                2.6313180923461914,
                0.21348094940185547,
                -6.365965366363525,
                0.591130793094635,
                4.148641586303711
            ],
            [
                -893.0125122070312,
                0.6328644156455994,
                5.223379611968994,
                6.164973735809326,
                6.097878932952881,
                3.0620410442352295,
                -0.796212911605835,
                -1.858128547668457,
                -0.7597164511680603,
                -0.6163774132728577,
                -5.2212018966674805,
                0.8604856133460999,
                3.8715980052948
            ],
            [
                -891.8641967773438,
                -4.334694862365723,
                2.8991262912750244,
                3.0012097358703613,
                2.5040454864501953,
                0.7696767449378967,
                0.3109185993671417,
                -3.9330925941467285,
                -5.649743556976318,
                -4.163054466247559,
                -3.9006943702697754,
                -1.9409058094024658,
                -2.323537588119507
            ],
            [
                -890.9918212890625,
                -0.7585126757621765,
                -1.0902214050292969,
                -3.119357109069824,
                -4.293031692504883,
                -2.2687253952026367,
                -0.0880429744720459,
                -1.2896761894226074,
                -0.8928248882293701,
                -4.964167594909668,
                -4.535434722900391,
                -0.7113205194473267,
                -4.683258533477783
            ],
            [
                -894.5484008789062,
                -5.478684425354004,
                -3.6322715282440186,
                -10.20645523071289,
                -7.916339874267578,
                -3.8653149604797363,
                -2.23716402053833,
                2.825047016143799,
                6.417963981628418,
                -4.58528995513916,
                -7.970461845397949,
                -2.56942081451416,
                -7.399971961975098
            ]])
test_input = test_input.reshape((1, no_frames, no_features))
test_output = model.predict(test_input, verbose=0)
print(test_output)



# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:42:48 2020

@author: akira
"""

import pandas as p
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsor=p.read_csv('entradas-breast.csv')
classe=p.read_csv('saidas-breast.csv')
classificador = Sequential()
classificador.add(Dense(units=8,activation='relu',kernel_initializer='normal',input_dim=30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=8,activation='relu',kernel_initializer='normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1,activation='sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(previsor,classe,batch_size=10,epochs=100)
classificador_json=classificador.to_json()
with open('classificador.json','w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador.h5')
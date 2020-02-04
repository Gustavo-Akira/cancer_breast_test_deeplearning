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
novo = np.array([[15,80,8.34,118,900,0.10,0.26,0.08,0.134,0.178,0.20,0.05,1090,0.07,4500,145,2,0.005,0.04,0.05,0.015,0.03,0.07,23,15,16.64,178.5,2018,0.14,0.185]])
previsao = classificador.predict(novo)
previsao=(previsao>0.5)
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:43:28 2020

@author: akira
"""

import pandas as pd
 
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
pr_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe,test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential();
classificador.add(Dense(units = 16,activation='relu',kernel_initializer = 'random_uniform',input_dim=30))
classificador.add(Dense(units = 16,activation='relu',kernel_initializer = 'random_uniform'))
classificador.add(Dense(units=1,activation = 'sigmoid'))
otimizador = keras.optimizers.Adam(lr=0.001, decay=0.001,clipvalue=0.5)
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(pr_treinamento,classe_treinamento,batch_size=10,epochs = 500)
pso0 = classificador.layers[0].get_weights()
pso1=classificador.layers[1].get_weights()
previsoes= classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precisap = accuracy_score(classe_teste,previsoes);
matriz = confusion_matrix(classe_teste,previsoes)
resultado = classificador.evaluate(previsores_teste,classe_teste)
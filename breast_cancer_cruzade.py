# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:02:15 2020

@author: akira
"""
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')
classificador = Sequential();
classificador.add(Dense(units = 8,activation='relu',kernel_initializer ='normal',input_dim=30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8,activation='relu',kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1,activation = 'sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

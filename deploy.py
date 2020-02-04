# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:42:48 2020

@author: akira
"""
import pandas as pd 
import numpy as np
from keras.models import model_from_json

arquivo  = open('classificador.json','r')
estrutura_rede=arquivo.read()
arquivo.close()
classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador.h5')
novo = np.array([[15,80,8.34,118,900,0.10,0.26,0.08,0.134,0.178,0.20,0.05,1090,0.07,4500,145,2,0.005,0.04,0.05,0.015,0.03,0.07,23,15,16.64,178.5,2018,0.14,0.185]])
previsao = classificador.predict(novo)
previsao=(previsao>0.5)
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')
classificador.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
resultado=classificador.evaluate(previsores,classe)
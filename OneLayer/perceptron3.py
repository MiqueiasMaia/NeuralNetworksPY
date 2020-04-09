# -*- coding: utf-8 -*-

import numpy as np

# entradas  = np.array([[0.3, 0.7], [1.5, 0.9], [0.4, 1], [1, 1.7]])
# saidas = np.array([0, 0, 0, 1]) #(E)
# entradas  = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# saidas = np.array([0, 1, 1, 1]) #(OR)
entradas  = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
saidas = np.array([0, 1, 1, 0]) #nunca irá conseguir ajustar (XOR)
pesos = np.array([0.0, 0.0])
aprendizagem = 0.1

def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

def calcSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

def treinar():
    erroTotal = 1
    while (erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calcSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] += (aprendizagem * entradas[i][j] * erro)
                print("Peso atualizado: "+str(pesos[j]))

        print("Total de erros: " +str(erroTotal))


treinar()

print("Rede neural treinada")
for i in range(len(saidas)):
    print(calcSaida(entradas[i]))

# Trabalho de Algebra linear computacional
# Breno
# Joao Pinheiro
'''
INPUTS do Programa (arquivo de entrada):
a) ordem N do sistema de equações
b) ICOD relativo ao método de análise
c) IDET = 0 não calcula determinante ou maior que 0 calcula o determinante
d) A matriz A e o vetor B
e) TOLm - tolerância máxima para a solução iterativa (qdo for o caso)
'''


# parsing inputs
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
import os
import pandas as pd
import argparse


# Trabalho 1
'''
Prepare um programa computacional (na linguagem de sua preferência) para efetuar a solução
de um sistema linear de equações AX = B onde o usuário possa escolher entre os métodos:
1. Decomposição LU (ICOD =1);
2. Decomposição de Cholesky (ICOD =2)
3. Procedimento iterativo Jacobi (ICOD =3) e
4. Procedimento iterativo Gauss-Seidel (ICOD =4).

Além disto, quando for requisitado pelo usuário e a técnica de solução permitir (caso contrário
deve ser emitido um “warning”), que seja efetuado o cálculo o determinante de A.

INPUTS do Programa (arquivo de entrada):
a) ordem N do sistema de equações
b) ICOD relativo ao método de análise
c) IDET = 0 não calcula determinante ou maior que 0 calcula o determinante
d) A matriz A e o vetor B
e) TOLm - tolerância máxima para a solução iterativa (qdo for o caso)
OUTPUTS do Programa (arquivo de saída):
a) Solução do sistema X;
b) Possíveis “erros de uso” (decomposições não são possíveis, possiblidade de não
convergência, etc.)
c) Determinante quando solicitado;
d) Número de iterações para convergência e histórico da variação do erro (TOL) da
solução nos casos dos métodos iterativos;

Obs.:
1) o programa deve ser desenvolvido visando o armazenamento mínimo de dados na
memória do computador (por exemplo, não deve ser criada uma nova matriz similar a
matriz A para a solução do sistema e equações);
2) não use rotinas prontas disponíveis na literatura/internet.
'''
# Lista de codigos de decomposicao
## Decomposiçao LU
## Decomposiçao de Cholesky
## Decomposiçao de Jacobi
## Decomposiçao de Gauss-Seidel


## Lista de metodos
# Metodo da potencia
# Metodos de jacobi

# Trabalho 3
'''
Prepare um programa computacional (na linguagem de sua preferência), dependendo da
escolha do usuário, para obter por interpolação (método Lagrange) ou regressão multilinear o
valor aproximado de uma função num determinando ponto.
1. Interpolação (ICOD =1);
2. Regressão (ICOD =2)
Além disto, quando for requisitado pelo usuário e a técnica de solução permitir (caso contrário
deve ser emitido um “warning”), que seja efetuado o cálculo o determinante de A.

INPUTS do Programa (arquivo de entrada):
a) ICOD relativo ao método de análise
b) N - número de pares de pontos (Xi, Yi)
c) X - coordenada do ponto que se deseja calcular o valor de y

OUTPUTS do Programa (arquivo de saída):
a) Valor de y estimado
b) Possíveis “erros de uso”;
Obs.: o programa deve ser desenvolvido visando o armazenamento mínimo de dados na
memória do computador
A entrega deverá conter:
1. Impressão dos arquivos com as rotinas desenvolvidas (todos juntos num mesmo pdf)
2. Um “pseudo” manual do usuário - orientações mínimas de como usar o programa e;
3. Um exemplo com dados de entrada e de saída
'''

# load data from file parsing inputs
'''
flags:
-im || --input-matriz : input_matriz 
-ir || --input-resultado : input_resultado
-TOLm || --tolerancia-maxima : tolerancia_maxima
-ICODD || --codigo-decomposicao : codigo_decomposicao
-ICODM || --codigo-metodo : codigo_metodo
-IDET || --determinante : determinante
-ordem || --ordem : ordem
'''

def main():
    parser = argparse.ArgumentParser(description='Programa de Algebra Linear')
    parser.add_argument('-im', '--input-matriz', type=str, help='input_matriz (arquivo de entrada) que representa a matriz A')
    parser.add_argument('-ir', '--input-resultado', type=str, help='input_resultado (arquivo de entrada) que representa o vetor B resultado da matriz A')
    parser.add_argument('-TOLm', '--tolerancia-maxima', type=float, help='tolerancia_maxima (default: 0.001) usado para o caso de iterativo')
    parser.add_argument('-ICOD', '--codigo-metodo', type=int, help='Codigo_metodo (1,2,3,4,5,6) qual metodo sera usado') 
    parser.add_argument('-IDET', '--determinante', type=bool, help='Determinante')
    parser.add_argument('-ordem', '--ordem', type=int, help='ordem da matriz quadrada')
    args = parser.parse_args()
    input_matriz = args.input_matriz
    input_resultado = args.input_resultado
    
    MatrizA =np.loadtxt(input_matriz, dtype=float, delimiter=' ')
    VetorB = np.loadtxt(input_resultado, dtype=float, delimiter=' ')

    print(MatrizA)
    if MatrizA.shape == (0,0):
        print("Matriz vazia")
    if MatrizA.shape[0] != MatrizA.shape[1]:
        print("Matriz não quadrada")
    if MatrizA.shape[0] != args.ordem:
        print("Matriz de ordem diferente da ordem passada")
    if VetorB.shape[0] < MatrizA.shape[0]:
        print("Vetor B de ordem diferente da ordem da matriz")
        print("Existem informaçoes faltando")

if __name__ == '__main__':
    main()
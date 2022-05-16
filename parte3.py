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

import numpy as np
from parte1 import metodo_iterativo_gauss_seidel
from parte1 import metodo_lu

def produtorio_phi(vetor_pts_x:np.array, cordenada_i: int,pt_x: float) -> float:
    produtorio = 1
    for k in range(len(vetor_pts_x)):
        if k == cordenada_i:
            continue
        produtorio *= (pt_x - vetor_pts_x[k]) / (vetor_pts_x[cordenada_i] - vetor_pts_x[k]) 
    return produtorio

    

def interpolacao_lagrange(vetor_pts_x:np.array, vetor_pts_y:np.array, pt_x: float) -> float:
    lista_auxiliar = []
    for i in range(len(vetor_pts_x)):
        lista_auxiliar.append(produtorio_phi(vetor_pts_x,i,pt_x))
    vetort_phi = np.array(lista_auxiliar)

    return np.dot(vetor_pts_y,vetort_phi)
    

def regressao_linear(vetor_x:np.array, vetor_y:np.array, pt_x:float) -> tuple([float,np.array]):
    matriz_P=np.stack((np.ones(vetor_x.size),vetor_x),axis=1)
    matriz_P_transpose=np.transpose(matriz_P)
    matriz_A=(np.matmul(matriz_P_transpose,matriz_P))
    print("matriz_A:\n",matriz_A)
    vetor_C=np.matmul(matriz_P_transpose,vetor_y)
    print("VetorC:\n",vetor_C)
    B=metodo_iterativo_gauss_seidel(matriz_A,vetor_C)[0]
    print("metodo_iterativo_gauss_seidel:\n",B)
    print("metodo_lu:\n",metodo_lu(matriz_A,vetor_C)[1])
    print("metodo burro:\n",np.matmul(np.linalg.inv(matriz_A),vetor_C))
    
    return (B[0]+B[1]*pt_x),B

def main():
    # vetor_teste_x=np.array([-2,0,1])
    # vetor_teste_y=np.array([-27,-1,0])    
    # print(interpolacao_lagrange(vetor_teste_x,vetor_teste_y,12.0))
    vetor_teste_x=np.array([1,2,3])
    vetor_teste_y=np.array([2,3.5,6.5])    
    print(regressao_linear(vetor_teste_x,vetor_teste_y,12.0))

if __name__ == '__main__':
    main()
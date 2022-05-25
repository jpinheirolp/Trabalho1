import argparse
from typing import Tuple
import numpy as np
from parte1 import metodo_lu

def produtorio_phi(vetor_pts_x:np.ndarray, cordenada_i: int,pt_x: float) -> float:
    produtorio = 1
    for k in range(vetor_pts_x.size):
        if k == cordenada_i:
            continue
        produtorio *= (pt_x - vetor_pts_x[k]) / (vetor_pts_x[cordenada_i] - vetor_pts_x[k]) 
    return produtorio

def interpolacao_lagrange(vetor_pts_x:np.ndarray, vetor_pts_y:np.ndarray, pt_x: float) -> float:
    saida = 0.0
    for i in range(vetor_pts_x.size):
        saida += vetor_pts_y[i] * produtorio_phi(vetor_pts_x,i,pt_x)
        
    return saida
    
def regressao_linear(vetor_x:np.ndarray, vetor_y:np.ndarray, pt_x:float, functions:list=[lambda x:1, lambda x:x]) -> Tuple[float,float,np.ndarray]:
    matriz_P = np.zeros((len(functions),vetor_x.shape[0]))
    
    for function_i in range(len(functions)):
        for x in range(vetor_x.size):
            matriz_P[function_i][x] = functions[function_i](vetor_x[x])
    matriz_P = np.transpose(matriz_P)
    matriz_P_transpose=np.transpose(matriz_P)
    matriz_A=(np.matmul(matriz_P_transpose,matriz_P))
    # print("matriz_A:\n",matriz_A)
    vetor_C=np.matmul(matriz_P_transpose,vetor_y)
    # print("VetorC:\n",vetor_C)
    B=metodo_lu(matriz_A,vetor_C)[1]
    # print("metodo_iterativo_gauss_seidel:\n",B)
    # print("metodo_lu:\n",metodo_lu(matriz_A,vetor_C)[1])
    # print("metodo burro:\n",np.matmul(np.linalg.inv(matriz_A),vetor_C))
    
    pt_y = 0
    for i in range(len(functions)):
        pt_y += functions[i](pt_x)*B[i]

    return pt_x,pt_y,B

def main():
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
    parser = argparse.ArgumentParser(description='Programa 3 de Algebra Linear')
    parser.add_argument('-mp', '--matriz_pontos', type=str, help='Matriz dos pontos')
    parser.add_argument('-ic', '--icod', type=int, help="ICOD relativo ao método de análise\n 1 - Interpolação\n 2 - Regressão")
    parser.add_argument('-x', '--x', type=float, help='Coordenada x de teste')

    args = parser.parse_args()

    matriz_pontos =np.loadtxt(args.matriz_pontos, dtype=float, delimiter=' ')
    if matriz_pontos.size == 0:
        print("Matriz de pontos vazia")
        return
    if matriz_pontos.shape[1] != 2:
        print("Matriz de pontos deve conter 2 colunas")
        return
    vetor_x = matriz_pontos[:,0]
    vetor_y = matriz_pontos[:,1]
    pt_x = args.x
    ICOD = args.icod
    if ICOD == 1:
        print("Interpolação")
        print("Pontos de entrada:\n",matriz_pontos)
        print("Valor estimado:", interpolacao_lagrange(vetor_x, vetor_y, pt_x))
    elif ICOD == 2:
        print("Regressão")
        print("Pontos de entrada:\n",matriz_pontos)
        ponto_testado, valor_estimado , vetor_A = regressao_linear(vetor_x, vetor_y, pt_x, [lambda x:1/np.exp(x), lambda x:np.log(x)])
        print("Valor testado:",ponto_testado)
        print("Valor estimado:", valor_estimado)
        print("Vetores de A:",vetor_A)
    else:
        print("ICOD inválido")

if __name__ == '__main__':
    main()
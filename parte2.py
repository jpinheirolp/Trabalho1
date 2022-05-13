# Trabalho 2
'''
Prepare um programa computacional (na linguagem de sua preferência) para calcular os
autovalores e autovetores de uma matriz A pelos métodos:
1. Método da Potência (ICOD =1);
2. Método de Jacobi (ICOD =2)
Além disto, quando for requisitado pelo usuário e a técnica de solução permitir (caso contrário
deve ser emitido um “warning”), que seja efetuado o cálculo o determinante de A.

INPUTS do Programa (arquivo de entrada):
a) a ordem N da matriz A (quadrada)
b) ICOD relativo ao método de análise
c) IDET - 0 não calcula determinante/ maior que 0 calcula o determinante
d) TOLm - tolerância máxima para a solução iterativa

OUTPUTS do Programa (arquivo de saída):
a) Autovalores e autovetores da matriz A;
b) Possíveis “erros de uso”;
c) Determinante quando solicitado;
d) Número de iterações para convergência.
Obs.: o programa deve ser desenvolvido visando o armazenamento mínimo de dados na
memória do computador

A entrega deverá conter:
1. Impressão dos arquivos com as rotinas desenvolvidas (todos juntos num mesmo pdf)
2. Um “pseudo” manual do usuário impresso - orientações mínimas de como usar o
programa e;
3. Um exemplo impresso com dados de entrada e de saída
'''
import numpy as np
import argparse


def metodo_potencia(matrizA: np.matrix, vetorX: np.array = np.array([]), TOLm:float = 0.0001) -> tuple([np.array,float]):
    # VetorX é o vetor x0 ate o x...
    
    # Passo 1 assumir um vetor inicial X0 como sendo um autovetor da solucao do problema
    # AX=yX e y0=1
    if vetorX.any(0) or vetorX.size == 0:
        # se nao tiver o vetorX ou se esse vetorX possuir algum 0 criar um vetor cheio de 1 do tamanho de matrizA
        vetorX=np.ones(shape=matrizA.shape[0])
    # Iniciando as variaveiss de calcular os passos 2-5 
    lambda_atual = 0
    lambda_anterior = np.linalg.norm(vetorX,ord=np.inf)
    residuo=np.inf
    numero_iteracoes = 0
    # Passo 2 calcular os problemas(passos 2-5) iterativamente alterando o vetor X0
    # condiçao de parada sendo a tolerancia maxima 10^-3 
    while residuo >= TOLm:
        numero_iteracoes += 1
        # Passo 3 calcular multiplicacao
        vetorX=np.matmul( vetorX, matrizA)
        # Passo 4 normalizar Y pelo maior valor ( norma infinita )
        # norma infinita do vetor x
        lambda_atual = float(np.linalg.norm(vetorX,np.inf,axis=0) )   
        vetorX = vetorX/lambda_atual
        # Passo 5 Calcular residuo
        residuo = np.abs( (lambda_atual - lambda_anterior) / lambda_atual )
        lambda_anterior = lambda_atual
    
    return vetorX,residuo, numero_iteracoes


def metodo_jacobi(matrizA,x):
    pass

# def main():
#     parser = argparse.ArgumentParser(description='Programa de Algebra Linear')
#     parser.add_argument('-im', '--input-matriz', type=str, help='input_matriz (arquivo de entrada) que representa a matriz A')
#     parser.add_argument('-TOLm', '--tolerancia-maxima', type=float, help='tolerancia_maxima (default: 0.001) usado para o caso de iterativo')
#     args = parser.parse_args()
#     input_matriz = args.input_matriz
    
#     # MatrizA =np.loadtxt(input_matriz, dtype=float, delimiter=' ')
#     #VetorB = np.loadtxt(input_resultado, dtype=float, delimiter=' ')

#     print(MatrizA)

MatrizA =np.loadtxt('matrizteste.txt', dtype=float, delimiter=' ')
print(metodo_potencia(MatrizA))

# if __name__ == '__main__':
#     main()
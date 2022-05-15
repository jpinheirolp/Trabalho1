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
import numpy as np

pivot = 0
elemento_matriz_l = 0

def metodo_lu(matriz_A:np.matrix,vetor_b:np.array) -> np.array: 
    n_linhas = matriz_A.shape[0]
    matriz_resultante = np.identity(n_linhas)
    for k in range(n_linhas - 1):
        pivot = matriz_A[k][k]
        
        for i in range(n_linhas - k -1):
            i += k + 1
            elemento_matriz_l = matriz_A[i][k] / pivot  
            matriz_resultante[i][k] = elemento_matriz_l    
            matriz_A[i][k] = elemento_matriz_l 
            print(matriz_resultante)
            for j in range(n_linhas - k - 1):
                j += k + 1
                matriz_A[i][j] -= matriz_A[k][j] * elemento_matriz_l
    return matriz_A

vetor_b = np.array([3,6,10])

matriz_A = np.loadtxt('matrizteste.txt', dtype=float, delimiter=' ')


print(metodo_lu(matriz_A, vetor_b))
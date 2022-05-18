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
from parte2 import metodo_potencia

pivot = 0
elemento_matriz_l = 0

def confere_diagonal_dominante(matriz:np.matrix) -> bool:
    n_linhas=matriz.shape[0]
    for i in range(n_linhas):
        somalinha = np.sum(np.absolute(matriz[i]))-matriz[i][i]
        if matriz[i][i] <= somalinha:
            return False
    for j in range(n_linhas):
        somalinha = np.sum(np.absolute(matriz[:][j]))-matriz[j][j]
        if matriz[j][j] <= somalinha:
            return False
    
    return True
  

def confere_positiva_definida(matriz:np.matrix) -> bool:
    # Calcular se todos os autovalores sao positivos
    if min(metodo_potencia(matriz)[0]) >= 0:
        return True
    return False

def substituicao_para_frente(matriz_A:np.matrix, vetor_B:np.array, method:str ) -> np.array:
    n_linhas=matriz_A.shape[0]
    ajustametodo=0 #lu
    if method == "choleski":
        ajustametodo= 1
    for i in range(n_linhas):
        vetor_B[i] = vetor_B[i] / (matriz_A[i][i] ** ajustametodo)
        for j in range(i):
            vetor_B[i] -= (vetor_B[j]*matriz_A[i][j]) / (matriz_A[i][i] ** ajustametodo)
    return vetor_B

def retrosubstituicao(matriz_A:np.matrix, vetor_B:np.array) -> np.array:
    n_linhas=matriz_A.shape[0]
    for i in range(n_linhas):
        i = n_linhas - i - 1
        vetor_B[i] = vetor_B[i] / matriz_A[i][i]

        for j in range(n_linhas - i - 1 ):
            j += i + 1
            vetor_B[i] -= (vetor_B[j] * matriz_A[i][j]) /  matriz_A[i][i]
        
    return vetor_B 

def metodo_lu(matriz_A:np.matrix,vetor_B:np.array) -> np.array: 
    n_linhas = matriz_A.shape[0]
    matriz_A = matriz_A.astype(float)
    vetor_B = vetor_B.astype(float)
    #fatoracao LU
    for k in range(n_linhas - 1):
        pivot = matriz_A[k][k]
        for i in range(n_linhas - k -1):
            i += k + 1
            elemento_matriz_l = matriz_A[i][k] / pivot  
            matriz_A[i][k] = elemento_matriz_l 
            
            for j in range(n_linhas - k - 1):
                j += k + 1
                matriz_A[i][j] -= matriz_A[k][j] * elemento_matriz_l

    #substituicao pra frente Ly = b
    # print("matriz lu\n",matriz_A)
    vetor_B = substituicao_para_frente(matriz_A,vetor_B,"cholesky")
    # print("vetor y\n",vetor_B)
    #retrosubstituicao Ux = y
    vetor_B = retrosubstituicao(matriz_A,vetor_B)
            
    return matriz_A, vetor_B
    
def metodo_cholesky(matriz_A:np.matrix,vetor_B:np.array) -> np.array:
    if not confere_positiva_definida:
        print("Nao e positiva definita")
        exit()
    n_linhas = matriz_A.shape[0]
    matriz_A = matriz_A.astype(float)
    vetor_B = vetor_B.astype(float) 
    for i in range(n_linhas):
        somatorio_i_square = 0
        for k in range(i):
            somatorio_i_square += (matriz_A[i][k])**2
        matriz_A[i][i] = (matriz_A[i][i] - somatorio_i_square)**(0.5)
      
        # print(i,i,matriz_A[i][i])
        for j in range(n_linhas - i - 1 ):
            j += i + 1
            somatorio_j_square = 0
              
            for k in range(i):
                somatorio_j_square += matriz_A[i][k]*matriz_A[j][k]
            
            matriz_A[j][i] = (matriz_A[i][j] - somatorio_j_square) / matriz_A[i][i]
            matriz_A[i][j] = matriz_A[j][i]

    # print(matriz_A)
    #substituicao pra frente Ly = b
    vetor_B = substituicao_para_frente(matriz_A,vetor_B,"lu")
    # print(vetor_B)
    #retrosubstituicao Ux = y
    vetor_B = retrosubstituicao(matriz_A,vetor_B)

    return matriz_A , vetor_B

def metodo_iterativo_jacobi(matriz_A: np.matrix,vetor_B: np.array , vetorX: np.array = np.array([]), TOLm:float = 0.0001) -> tuple([np.array,float,int]):
    # VetorX é o vetor x0 ate o x...
    if not confere_diagonal_dominante(matriz_A):
        print("Nao e diagonal dominante")
        exit()
    n_linhas = matriz_A.shape[0]
    matriz_A = matriz_A.astype(float)
    vetorX = vetorX.astype(float)
    if vetorX.any(0) or vetorX.size == 0:
        vetorX=np.ones(shape=matriz_A.shape[0])

    residuo=np.inf
    numero_iteracoes = 0
    elemento_linha = 0

    while (residuo >= TOLm):
        vetorX_atualizado = np.ones(shape=matriz_A.shape[0],dtype=float)
        numero_iteracoes += 1
        for i in range(n_linhas):
            elemento_linha = vetorX[i]
            vetorX[i] = 0
            vetorX_atualizado[i] = (vetor_B[i] - np.dot(vetorX, matriz_A[i]))/matriz_A[i][i]
            vetorX[i] = elemento_linha
            # print(vetorX_atualizado)

        diferenca = vetorX_atualizado - vetorX
    
        # Passo 5 Calcular residuo
        residuo = np.abs(float(np.linalg.norm(diferenca,2,axis=0)) / float(np.linalg.norm(vetorX_atualizado,2,axis=0)))
        vetorX = vetorX_atualizado
    
    return vetorX, residuo, numero_iteracoes  

def metodo_iterativo_gauss_seidel(matriz_A: np.matrix,vetor_B: np.array , vetorX: np.array = np.array([]), TOLm:float = 0.0001) -> tuple([np.array,float,int]):
    if not confere_diagonal_dominante(matriz_A):
        print("Nao e diagonal dominante")
        exit()
    # VetorX é o vetor x0 ate o x...
    n_linhas = matriz_A.shape[0]
    matriz_A = matriz_A.astype(float)
    vetorX = vetorX.astype(float)
    if vetorX.any(0) or vetorX.size == 0:
        vetorX=np.ones(shape=matriz_A.shape[0])

    residuo=np.inf
    numero_iteracoes = 0

    while (residuo >= TOLm):
        vetorX_velho = np.copy(vetorX)
        numero_iteracoes += 1
        for i in range(n_linhas):
            vetorX[i] = 0
            vetorX[i] = (vetor_B[i] - np.dot(vetorX, matriz_A[i]))/matriz_A[i][i]
        
        diferenca =  vetorX - vetorX_velho
    
        # Passo 5 Calcular residuo
        residuo = np.abs(float(np.linalg.norm(diferenca,2,axis=0)) / float(np.linalg.norm(vetorX,2,axis=0)))
    
    return vetorX, residuo, numero_iteracoes

# vetor_B = np.array([0.6,-0.3,-0.6])
def main():
    matriz_A = np.loadtxt('matrizteste.txt', dtype=float, delimiter=' ')
    vetor_B = np.loadtxt('vetorteste.txt', dtype=float, delimiter=' ')

    print(metodo_iterativo_jacobi(matriz_A, vetor_B))
    print(metodo_iterativo_gauss_seidel(matriz_A, vetor_B))

if __name__ == '__main__':
    main()
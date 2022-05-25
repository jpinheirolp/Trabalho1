import functools
import numpy as np
import argparse


def confere_matriz_quadrada(matriz:np.matrix) -> bool:
    if matriz.shape[0] == matriz.shape[1]:
        return True
    return False

def confere_simetria(matriz: np.matrix) -> bool:
    if not confere_matriz_quadrada(matriz):
        return False
    for linha in range(matriz.shape[0]): 
        for col in range(matriz.shape[1] - linha):
            col += linha
           
            if matriz[linha][col] != matriz[col][linha]: # vamo testar essa func
                return False
    return True   

def metodo_potencia(matriz_A: np.matrix, vetorX: np.array = np.array([]), TOLm: float = 0.0001) -> tuple([np.array,list,int]):
    # VetorX é o vetor x0 ate o x...
    
    # Passo 1 assumir um vetor inicial X0 como sendo um autovetor da solucao do problema
    # AX=yX e y0=1
    if vetorX.any(0) or vetorX.size == 0:
        # se nao tiver o vetorX ou se esse vetorX possuir algum 0 criar um vetor cheio de 1 do tamanho de matriz_A
        vetorX=np.ones(shape=matriz_A.shape[0])
    # Iniciando as variaveiss de calcular os passos 2-5 
    lambda_atual = 0
    lambda_anterior = np.linalg.norm(vetorX,ord=np.inf)
    residuo=1
    numero_iteracoes = 0
    historico_residuo = []
    # Passo 2 calcular os problemas(passos 2-5) iterativamente alterando o vetor X0
    # condicao de parada sendo a tolerancia maxima 10^-3
    while (residuo >= TOLm):
        numero_iteracoes += 1
        # Passo 3 calcular multiplicacao
        vetorX=np.matmul(vetorX, matriz_A)
        # Passo 4 normalizar Y pelo maior valor ( norma infinita )
        # norma infinita do vetor x
        lambda_atual = float(np.linalg.norm(vetorX,ord=np.inf,axis=0))   
        vetorX = vetorX/lambda_atual
        # Passo 5 Calcular residuo
        residuo = np.abs((lambda_atual - lambda_anterior) / lambda_atual)
        lambda_anterior = lambda_atual
        historico_residuo.append(residuo)
    
    return vetorX, historico_residuo, numero_iteracoes , lambda_atual 

def get_maior_elemento_matriz_simetrica(matriz_A: np.matrix) -> tuple([float,int,int]):
    if not confere_matriz_quadrada(matriz_A):
        return False
    maior_elemento = 0
    linha_maior_elemento = 0
    coluna_maior_elemento = 0

    for linha in range(matriz_A.shape[0]): 
        for col in range(matriz_A.shape[1]):
            if linha == col:
                continue
            if abs(matriz_A[linha][col]) > abs(maior_elemento): # vamo testar essa func
                maior_elemento = matriz_A[linha][col] 
                linha_maior_elemento = linha
                coluna_maior_elemento = col

    return maior_elemento, linha_maior_elemento, coluna_maior_elemento       

def constroi_matriz_rotacao(nlinhas: int,i: int,j: int,Aij: float,Aii: float,Ajj: float) -> np.matrix:
    matriz_resultante = np.identity(nlinhas)
    angulo_rotacao = 0
    if Aii == Ajj:
        angulo_rotacao = np.pi/4.0
    else:
        angulo_rotacao = np.arctan( (2*Aij) / (Aii-Ajj) ) / 2

    matriz_resultante[i][i] =  np.cos(angulo_rotacao)
    matriz_resultante[i][j] =  np.sin(angulo_rotacao)
    matriz_resultante[j][i] = -np.sin(angulo_rotacao)
    matriz_resultante[j][j] =  np.cos(angulo_rotacao)

    return matriz_resultante

def metodo_jacobi(matriz_A:np.matrix,TOLm:float = 0.0001) -> tuple([np.matrix,np.matrix,int]):
    if not confere_simetria(matriz_A):
        print("Error: Nao e possivel usar jacobi em matriz nao simetrica")
        exit()
    nlinhas = matriz_A.shape[0]
    #passo 1 primeiro comeca como identidade depois a matrizx e alterada
    matriz_x = np.identity(nlinhas)
    print(TOLm)
    maior_elemento=np.inf
    numero_iteracoes=0
    historico_residuo = []
    #passo 2
    while(maior_elemento > TOLm):
        #passo 2.1
        maior_elemento, i, j = get_maior_elemento_matriz_simetrica(matriz_A)
        #passo 2.2 
        matriz_p = constroi_matriz_rotacao(nlinhas,i,j, matriz_A[i][j], matriz_A[i][i], matriz_A[j][j])
        #passo 2.3
        matriz_A = np.matmul(matriz_p.T,matriz_A)
        matriz_A = np.matmul(matriz_A,matriz_p)
        matriz_x = np.matmul(matriz_x,matriz_p)
        numero_iteracoes+=1
        historico_residuo.append(maior_elemento)
    
    return matriz_A.diagonal(), matriz_x, historico_residuo, numero_iteracoes
    
def main():
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
    parser = argparse.ArgumentParser(description='Programa 2 de Algebra Linear')
    parser.add_argument('-im', '--matriz_A', type=str, help='Matriz A',required=True)
    parser.add_argument('-ib', '--vetor_B', type=str, help='Vetor B',required=True)
    parser.add_argument('-ic', '--icod', type=int, help='ICOD relativo ao método de análise\n 1 - Potencia\n 2 - Jacobi',required=True)
    parser.add_argument('-id', '--idet', type=int, help='IDET',required=True)
    parser.add_argument('-it', '--tol', type=float, help='TOLm',default=0.0001)
    
    args = parser.parse_args()

    MatrizA =np.loadtxt(args.matriz_A, dtype=float, delimiter=' ')
    VetorB =np.loadtxt(args.vetor_B, dtype=float, delimiter=' ')

    TOLm = args.tol
    ICOD = args.icod
    IDET = args.idet
    
    print(get_maior_elemento_matriz_simetrica(MatrizA))
    if ICOD == 1:
        print("Metodo da Potencia")
        print("Matriz A:\n",MatrizA)
        autovetor_potencia, historico_residuo, numero_iteracoes, maiorautovalor = metodo_potencia(matriz_A=MatrizA, TOLm=TOLm)
        print("Autovetor:", autovetor_potencia)
        print("Maior Autovalor:", maiorautovalor)
        print("Residuo:", historico_residuo)
        print("Numero de Iteracoes:", numero_iteracoes)
        if IDET:
            print("Nao e possivel calcular o determinante pelo metodo da potencia")
            
    elif ICOD == 2:
        print("Metodo de Jacobi")
        print("Matriz A:\n", MatrizA)
        autovalores_jacobi, matriz_jacobi, historico_residuo, numero_iteracoes = metodo_jacobi(matriz_A=MatrizA, TOLm=TOLm)
        print("Matriz Autovetores:\n", matriz_jacobi)
        print("Autovalores: ", autovalores_jacobi)
        print("Residuo:", historico_residuo)
        print("Numero de Iteracoes: ", numero_iteracoes)
        if IDET:
            print("Determinante:",functools.reduce(lambda x,y: x*y, autovalores_jacobi))
            print("det(A):",np.linalg.det(MatrizA) )
    else:
        print("Codigo de Metodo Invalido")
        exit()

if __name__ == '__main__':
    main()
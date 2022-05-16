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


def confere_matriz_quadrada(matriz:np.matrix) -> bool:
    if matriz.shape[0] == matriz.shape[1]:
        return True
    return False

def confere_simetria( matriz: np.matrix ) -> bool:
    if not confere_matriz_quadrada(matriz_A):
        return False
    for linha in range(matriz.shape[0]): 
        for col in range(matriz.shape[1] - linha):
            col += linha
           
            if matriz[linha][col] != matriz[col][linha]: # vamo testar essa func
                return False
    return True   

def metodo_potencia(matriz_A: np.matrix, vetorX: np.array = np.array([]), TOLm:float = 0.0001) -> tuple([np.array,float,int]):
    # VetorX é o vetor x0 ate o x...
    
    # Passo 1 assumir um vetor inicial X0 como sendo um autovetor da solucao do problema
    # AX=yX e y0=1
    if vetorX.any(0) or vetorX.size == 0:
        # se nao tiver o vetorX ou se esse vetorX possuir algum 0 criar um vetor cheio de 1 do tamanho de matriz_A
        vetorX=np.ones(shape=matriz_A.shape[0])
    # Iniciando as variaveiss de calcular os passos 2-5 
    lambda_atual = 0
    lambda_anterior = np.linalg.norm(vetorX,ord=np.inf)
    residuo=np.inf
    numero_iteracoes = 0
    # Passo 2 calcular os problemas(passos 2-5) iterativamente alterando o vetor X0
    # condicao de parada sendo a tolerancia maxima 10^-3 
    while (residuo >= TOLm) and (numero_iteracoes <= 100):
        numero_iteracoes += 1
        # Passo 3 calcular multiplicacao
        vetorX=np.matmul(vetorX, matriz_A)
        # Passo 4 normalizar Y pelo maior valor ( norma infinita )
        # norma infinita do vetor x
        lambda_atual = float(np.linalg.norm(vetorX,np.inf,axis=0))   
        vetorX = vetorX/lambda_atual
        # Passo 5 Calcular residuo
        residuo = np.abs((lambda_atual - lambda_anterior) / lambda_atual)
        lambda_anterior = lambda_atual
    
    return vetorX, residuo, numero_iteracoes  

def get_maior_elemento_matriz_simetrica(matriz_A: np.matrix) -> tuple([float,int,int]):
    if not confere_matriz_quadrada(matriz_A):
        return False
    maior_elemento = 0
    linha_maior_elemento = 0
    coluna_maior_elemento = 0

    for linha in range(matriz_A.shape[0]): 
        for col in range(matriz_A.shape[1] - linha - 1):
            col += linha + 1
            
            if abs(matriz_A[linha][col]) > abs(maior_elemento): # vamo testar essa func
                maior_elemento = matriz_A[linha][col] 
                linha_maior_elemento = linha
                coluna_maior_elemento = col

    return maior_elemento, linha_maior_elemento, coluna_maior_elemento       

def constroi_matriz_rotacao( nlinhas:int, i:int, j:int, Aij:float, Aii:float, Ajj:float ) -> np.matrix:
    matriz_resultante = np.identity(nlinhas)
    angulo_rotacao = 0
    if Aii == Ajj:
        angulo_rotacao = np.pi/4.0
    else:
        angulo_rotacao = np.arctan( (2*Aij) / (Aii-Ajj) ) / 2

    matriz_resultante[i][i] =  np.cos( angulo_rotacao )
    matriz_resultante[i][j] =  np.sin( angulo_rotacao )
    matriz_resultante[j][i] = -np.sin( angulo_rotacao )
    matriz_resultante[j][j] =  np.cos( angulo_rotacao )

    return matriz_resultante

def metodo_jacobi(matriz_A:np.matrix,TOLm:float = 0.0001) -> tuple([np.matrix,np.matrix,int]):
    if not confere_simetria(matriz_A):
        print("Error: Nao e possivel usar jacobi em matriz nao simetrica")
        exit()
        
    nlinhas = matriz_A.shape[0]
    #passo 1 primeiro comeca como identidade depois a matrizx e alterada
    matriz_x = np.identity(nlinhas)
    
    maior_elemento=np.inf
    numero_iteracoes=0
    #passo 2
    while(maior_elemento > TOLm) and (numero_iteracoes <= 100):
        #passo 2.1
        maior_elemento, i, j = get_maior_elemento_matriz_simetrica(matriz_A)
        #passo 2.2 
        matriz_p = constroi_matriz_rotacao(nlinhas,i,j, matriz_A[i][j], matriz_A[i][i], matriz_A[j][j])
        # print("p\n",matriz_p)
        #passo 2.3
        matriz_A = np.matmul(matriz_p.T,matriz_A)
        #print(matriz_A,"\n")
        matriz_A = np.matmul(matriz_A,matriz_p)
        #print(matriz_A,"\n")
        matriz_x = np.matmul(matriz_x,matriz_p)
        #print(matriz_x,"\n")
        numero_iteracoes+=1
    
    return matriz_x.diagonal(), matriz_A, numero_iteracoes
    



def main():
#     parser = argparse.ArgumentParser(description='Programa de Algebra Linear')
#     parser.add_argument('-im', '--input-matriz', type=str, help='input_matriz (arquivo de entrada) que representa a matriz A')
#     parser.add_argument('-TOLm', '--tolerancia-maxima', type=float, help='tolerancia_maxima (default: 0.001) usado para o caso de iterativo')
#     args = parser.parse_args()
#     input_matriz = args.input_matriz
    
#     # MatrizA =np.loadtxt(input_matriz, dtype=float, delimiter=' ')
#     #VetorB = np.loadtxt(input_resultado, dtype=float, delimiter=' ')

#     print(MatrizA)

    matriz_A =np.loadtxt('matrizteste.txt', dtype=float, delimiter=' ')
    #print(metodo_potencia(matriz_A))
    # print(matriz_A)
    metodo,jacobi,null = metodo_jacobi(matriz_A)
    print(metodo,"\n\n",jacobi)

if __name__ == '__main__':
    main()
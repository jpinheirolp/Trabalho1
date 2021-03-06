import argparse
from typing import Tuple
import numpy as np
import functools
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
    for i in metodo_potencia(matriz)[0]:
        if i < 0:
            return False
    return True

def substituicao_para_frente(matriz_A:np.matrix, vetor_B:np.ndarray, method:str ) -> np.ndarray:
    n_linhas=matriz_A.shape[0]
    ajustametodo=0 #lu
    if method == "choleski":
        ajustametodo= 1
    for i in range(n_linhas):
        vetor_B[i] = vetor_B[i] / (matriz_A[i][i] ** ajustametodo)
        for j in range(i):
            vetor_B[i] -= (vetor_B[j]*matriz_A[i][j]) / (matriz_A[i][i] ** ajustametodo)
    return vetor_B

def retrosubstituicao(matriz_A:np.matrix, vetor_B:np.ndarray) -> np.ndarray:
    n_linhas=matriz_A.shape[0]
    for i in range(n_linhas):
        i = n_linhas - i - 1
        vetor_B[i] = vetor_B[i] / matriz_A[i][i]

        for j in range(n_linhas - i - 1 ):
            j += i + 1
            vetor_B[i] -= (vetor_B[j] * matriz_A[i][j]) /  matriz_A[i][i]
        
    return vetor_B 

def metodo_lu(matriz_A:np.matrix,vetor_B:np.ndarray) -> Tuple[np.matrix,np.ndarray]: 
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
    vetor_B = substituicao_para_frente(matriz_A,vetor_B,"lu")
    # print("vetor y\n",vetor_B)
    #retrosubstituicao Ux = y
    vetor_B = retrosubstituicao(matriz_A,vetor_B)
            
    return matriz_A, vetor_B
    
def metodo_cholesky(matriz_A:np.matrix,vetor_B:np.ndarray) -> Tuple[np.matrix,np.ndarray]:
    if not confere_positiva_definida(matriz_A):
        print("Error: n??o ?? possivel executar pois Nao e positiva definida")
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
    vetor_B = substituicao_para_frente(matriz_A,vetor_B,"cholesky")
    # print(vetor_B)
    #retrosubstituicao Ux = y
    vetor_B = retrosubstituicao(matriz_A,vetor_B)

    return matriz_A , vetor_B

def metodo_iterativo_jacobi(matriz_A: np.matrix,vetor_B: np.ndarray , vetorX: np.ndarray = np.array([]), TOLm:float = 0.0001) -> Tuple[np.ndarray,list,int]:
    # VetorX ?? o vetor x0 ate o x...
    if not confere_diagonal_dominante(matriz_A):
        print("Error: n??o ?? possivel executar quando diagonal nao e dominante")
        exit()
    n_linhas = matriz_A.shape[0]
    matriz_A = matriz_A.astype(float)
    vetorX = vetorX.astype(float)
    historico_residuo=[]
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
        historico_residuo.append(residuo)

    return vetorX, historico_residuo, numero_iteracoes  

def metodo_iterativo_gauss_seidel(matriz_A: np.matrix,vetor_B: np.ndarray , vetorX: np.ndarray = np.array([]), TOLm:float = 0.0001) -> Tuple[np.ndarray,list,int]:
    if not confere_diagonal_dominante(matriz_A):
        print("Error: n??o ?? possivel executar quando diagonal nao e dominante")
        exit()
    # VetorX ?? o vetor x0 ate o x...
    historico_residuo=[]
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
        historico_residuo.append(residuo)
    
    return vetorX, historico_residuo, numero_iteracoes

def main():
    # Trabalho 1
    '''
    Prepare um programa computacional (na linguagem de sua prefer??ncia) para efetuar a solu????o
    de um sistema linear de equa????es AX = B onde o usu??rio possa escolher entre os m??todos:
    1. Decomposi????o LU (ICOD =1);
    2. Decomposi????o de Cholesky (ICOD =2)
    3. Procedimento iterativo Jacobi (ICOD =3) e
    4. Procedimento iterativo Gauss-Seidel (ICOD =4).

    Al??m disto, quando for requisitado pelo usu??rio e a t??cnica de solu????o permitir (caso contr??rio
    deve ser emitido um ???warning???), que seja efetuado o c??lculo o determinante de A.

    INPUTS do Programa (arquivo de entrada):
    a) ordem N do sistema de equa????es
    b) ICOD relativo ao m??todo de an??lise
    c) IDET = 0 n??o calcula determinante ou maior que 0 calcula o determinante
    d) A matriz A e o vetor B
    e) TOLm - toler??ncia m??xima para a solu????o iterativa (qdo for o caso)

    OUTPUTS do Programa (arquivo de sa??da):
    a) Solu????o do sistema X;
    b) Poss??veis ???erros de uso??? (decomposi????es n??o s??o poss??veis, possiblidade de n??o
    converg??ncia, etc.)
    c) Determinante quando solicitado;
    d) N??mero de itera????es para converg??ncia e hist??rico da varia????o do erro (TOL) da
    solu????o nos casos dos m??todos iterativos;

    Obs.:
    1) o programa deve ser desenvolvido visando o armazenamento m??nimo de dados na
    mem??ria do computador (por exemplo, n??o deve ser criada uma nova matriz similar a
    matriz A para a solu????o do sistema e equa????es);
    2) n??o use rotinas prontas dispon??veis na literatura/internet.
    '''
    parser = argparse.ArgumentParser(description='Programa 1 de Algebra Linear')
    parser.add_argument('-im', '--matriz_A', type=str, help='Matriz A',required=True)
    parser.add_argument('-ib', '--vetor_B', type=str, help='Vetor B',required=True)
    parser.add_argument('-ic', '--icod', type=int, help='ICOD relativo ao m??todo de an??lise; 1 - LU; 2 - Cholesky; 3 - Iterativo Jacobi;, 4 - Iterativo Gauss-Seidel',required=True)
    parser.add_argument('-id', '--idet', type=int, help='IDET',default=0)
    parser.add_argument('-it', '--tol', type=float, help='TOLm',default=0.0001)
    args = parser.parse_args()

    TOLm = args.tol
    ICOD = args.icod
    IDET = args.idet

    matriz_A = np.loadtxt(args.matriz_A, dtype=float, delimiter=' ')
    vetor_B = np.loadtxt(args.vetor_B, dtype=float, delimiter=' ')
   
    if ICOD == 1:
        print("Decomposicao LU")
        matriz_lu, vetorX = metodo_lu(matriz_A, vetor_B)
        print("Matriz LU:\n", matriz_lu)
        print("Resposta",vetorX) # resposta
        if IDET:  
            print("Determinante",functools.reduce(lambda x,y: x*y, matriz_lu.diagonal()))
    elif ICOD == 2:
        print("Decomposicao Cholesky")
        matriz_cholesky, vetorX = metodo_cholesky(matriz_A, vetor_B)
        print("Resposta:",vetorX)
        print("Matriz Cholesky:\n", matriz_cholesky)
        if IDET:  
            print("Determinante",(functools.reduce(lambda x,y: x*y, matriz_cholesky.diagonal())**2))
    elif ICOD == 3:
        print("Metodo iterativo Jacobi")
        vetorX, historico_residuo, numero_iteracoes = metodo_iterativo_jacobi(matriz_A, vetor_B, TOLm=TOLm )
        print("Vetor Solucao",vetorX)
        print("Historico residuo",historico_residuo)
        print("N de iteracoes",numero_iteracoes)
        if IDET:  
            print("Nao e possivel fazer determinante de maneira otimizada por jacobi")
    elif ICOD == 4:
        print("Metodo iterativo Gauss Seidel")
        vetorX, historico_residuo, numero_iteracoes = metodo_iterativo_gauss_seidel(matriz_A, vetor_B, TOLm=TOLm)
        print("Vetor Solucao:",vetorX)
        print("Historico residuo:",historico_residuo)
        print("N de iteracoes:",numero_iteracoes)
        if IDET:  
            # calcula e printa determinante via gauss-seidel
            print("Nao e possivel fazer determinante de maneira otimizada por gauss seidel")
    else:
        print("Error: ICOD n??o definido")
        exit()

if __name__ == '__main__':
    main()
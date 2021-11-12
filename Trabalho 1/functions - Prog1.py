import cv2 as cv
import numpy as np

#Questão 2.1: Função que reduz e inteporla a imagema
def dec_int(img, n):
    height, width = img.shape[:2] #Largura e Altura da imagem

    #Nova Largura e Altura da imagem
    newWidth =  int(width/n)
    newHeight = int (height/n)

    resized = np.zeros([newHeight, newWidth, 3], dtype = np.uint8) #Iniciando matriz da imagem redimensionada

    #Redimensionando
    for i in range(newHeight):
        for j in range(newWidth):
            resized[i][j] = img[i*n][j*n]

    interpolated = np.zeros([newHeight*n, newWidth*n, 3], dtype = np.uint8) #Iniciando matriz da imagem interpolada

    #Interpolando
    for i in range(newHeight*n):
        for j in range(newWidth*n):
            interpolated[i][j] = resized[i//n][j//n]
    
    return resized, interpolated


#Questão 2.2: Função que aprimora a imagem
def egde_improv(img): #Transformação Logarítmica
    
    improved = np.array(np.max(img)*(img/np.max(img))**2, dtype = np.uint8)   

    return improved


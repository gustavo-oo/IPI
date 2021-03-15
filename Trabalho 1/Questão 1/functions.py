import cv2 as cv
import numpy as np
import math

#Função que reduz e inteporla a imagem
def dec_int(img, n):
    width, height = img.shape[:2] #Largura e Altura da imagem

    #Nova Largura e Altura da imagem
    newWidth =  int(width/n)
    newHeight = int (height/n)

    resized = np.zeros([newWidth, newHeight, 3], dtype = np.uint8) #Iniciando matriz da imagem redimensionada

    #Redimensionando
    for i in range(newWidth):
        for j in range(newHeight):
            resized[i][j] = img[i*n][j*n]

    interpolated = np.zeros([newWidth*n, newHeight*n, 3], dtype = np.uint8) #Iniciando matriz da imagem interpolada

    #Interpolando
    for i in range(newWidth*n):
        for j in range(newHeight*n):
            interpolated[i][j] = resized[i//n][j//n]
    
    return interpolated

#Função dec_int aprimorada
def dec_int_improv(img, n):
    width, height = img.shape[:2] #Largura e Altura da imagem

    #Nova Largura e Altura da imagem
    newWidth =  int(width/n)
    newHeight = int (height/n)

    resized = np.zeros([newWidth, newHeight, 3], dtype = np.uint8) #Iniciando matriz da imagem redimensionada

    #Redimensionando
    for i in range(newWidth):
        for j in range(newHeight):
            resized[i][j] = img[i*n][j*n]

    interpolated = np.zeros([newWidth*n, newHeight*n, 3], dtype = np.uint8) #Iniciando matriz da imagem interpolada

    #Interpolando
    for i in range(0, newWidth*n, n):
        for j in range(0, newHeight*n, n):
            media = resized[i//n][j//n]
            if(i//n < newWidth - 1):
                media += resized[i//n + 1][j//n]
                media = media/2
            if(i//n < newHeight - 1):
                media += resized[i//n][j//n + 1]
                media = media/2
            if(i//n < newWidth - 1 and i//n < newHeight - 1):
                media += resized[i//n + 1][j//n + 1]
                media = media/4

            interpolated[i][j] = resized[i//n][j//n]
            for l in range(i//n + 1, i):
                for m in range(j//n + 1, j):
                    interpolated[l][m] = media

    return interpolated


#Função que aprimora a imagem
def egde_improv(img): #Transformação Logarítmica
    #Transformação gamma = 3
    improved = np.array(255*(img/255)**2, dtype = np.uint8)   

    # c = 255 / np.log(1 + np.max(img)) 
    # improved = c * (np.log(img + 1)) 
    # improved = np.array(log_image, dtype = np.uint8) 
  

    cv.imshow('improved', improved)
    cv.imshow('original', img)
    cv.waitKey(0)


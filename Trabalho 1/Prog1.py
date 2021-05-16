import cv2 as cv
import numpy as np

from functions import dec_int, egde_improv

#Leitura da imagem e definição do fator de redução
img = cv.imread('test80.jpg')
n = 6

#Extração de altura e largura da imagem para as funções de redução e interpolação cúbica
height, width = img.shape[:2]
newWidth = int(width/n)
newHeight = int(height/n)

#Redução e interpolação da imagem utilizando a função dec_int
resized, interpolated = dec_int(img, n)

#Redução e interpolação cúbica da imagem utilizando a função cv.resize
cubic_resized = cv.resize(img, (newWidth, newHeight), interpolation = cv.INTER_CUBIC)
cubic_interpolated = cv.resize(cubic_resized, (width, height), interpolation = cv.INTER_CUBIC)

#Aprimoramento da imagem atráves da função egde_improv
interpolated_improv = egde_improv(interpolated)
cubic_interpolated_improv = egde_improv(cubic_interpolated)

#Mostrando resultados obtidos
cv.imshow('resized', resized)
cv.imshow('interpolated', interpolated)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow('cubic_resized', cubic_resized)
cv.imshow('cubic_interpolated', cubic_interpolated)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow('interpolated_improved', interpolated_improv)
cv.imshow('cubic_improved', cubic_interpolated_improv)

cv.waitKey(0)
cv.destroyAllWindows()
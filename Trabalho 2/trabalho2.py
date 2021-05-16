import cv2 as cv
import numpy as np

#Função que binariza uma imagem dado um limiar
def binarize(img, limiar):
    img_binary = cv.threshold(img, limiar, 255, cv.THRESH_BINARY)[1]
    return img_binary

# ========================================== Questão 1 ========================================== #
#Lendo a imagem e convertendo para escala de cinza
img = cv.imread('morf_test.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Imagem Binazarizada Diretamente
img_binary = binarize(img, 170)

#Algoritmo que reduz variações no fundo
kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
inverted = cv.bitwise_not(img_gray)
tophat_img = cv.morphologyEx(inverted, cv.MORPH_TOPHAT, kernel)
tophat_img = cv.bitwise_not(tophat_img)
tophat_img = binarize(tophat_img, 220)

#Resultados
cv.imshow('Binarizacao Direta', img_binary)
cv.imshow("Imagem com Transformacao Tophat", tophat_img)
cv.waitKey(0)
cv.destroyAllWindows()


# ========================================== Questão 2 ========================================== #
#Lendo a imagem
img = cv.imread("cookies.tif")

#Binarizando a imagem
binary_img = binarize(img, 100)

#Eliminando Cookie mordido
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (118, 118))
mask = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

#Retornando Cookie não mordido ao tamanho normal
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
mask_corrected = cv.dilate(mask, kernel)
mask_corrected = np.bitwise_and(mask_corrected, binary_img)

#Imagem Final
final = np.bitwise_and(img, mask_corrected)

#Resultados
cv.imshow("Imagem Binarizada", binary_img)
cv.imshow('Cookie mordido removido', mask)
cv.imshow('Cookie nao mordido corrigido', mask_corrected)
cv.imshow('Imagem Final', final)
cv.waitKey(0)
cv.destroyAllWindows()


# ========================================== Questão 3 ========================================== #
#Lendo a imagem e convertendo para escala de cinza
img = cv.imread('img_cells.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Binarizando a  Imagem
binary_img = binarize(img_gray, 150)

#Algoritmo para preenchimentos de buracos
inverted = cv.bitwise_not(binary_img)
contours = cv.findContours(inverted, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)[0]
for contour in contours:
    cv.drawContours(inverted, [contour], 0, 255, cv.FILLED)

# Remoção de alguns ruídos na imagem
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
img_noise_removed = cv.morphologyEx(inverted, cv.MORPH_OPEN,kernel, iterations = 2)

# Encontrando a região onde podem haver celulas e parte do fundo
background = cv.dilate(img_noise_removed, kernel, iterations= 3)

# Função de distancia para encontrar a região onde certamente possuem celulas
dist_transform = cv.distanceTransform(img_noise_removed, cv.DIST_L2, 5)
foreground = cv.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)[1]
foreground = np.uint8(foreground)

# Região da borda entre as celulas e o fundo, não havendo certeza de onde começa um e termina o outro
border = cv.subtract(background, foreground)

# Função que cria marcadores positivos para regiões das células e 0 para a regiao restante
markers = cv.connectedComponents(foreground)[1]

# Na função watersheed, a regiao de borda deve ser 0, logo somamos 1 para nao ocorrer conflito
markers = markers+1

# Adicionamos os marcadores da região de borda como 0
markers[border==255] = 0

#Segmentação Watershed
markers = cv.watershed(img, markers)

#Marcação na Imagem com o resultado do watersheed
img[markers == -1] = [0, 0, 255]

#Resultados
cv.imshow("Imagem Binarizada", binary_img)
cv.imshow("Imagem com Buracos Preeenchidos", inverted)
cv.imshow("Imagem com Ruidos Removidos", img_noise_removed)
cv.imshow("Imagem da Regiao de Fundo", background)
cv.imshow("Imagem da Regiao das Celulas", foreground)
cv.imshow("Imagem da Regiao de borda", border)
cv.imshow("Imagem Final com Marcadores", img)
cv.waitKey(0)
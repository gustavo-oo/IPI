import cv2 as cv
import numpy as np

#Funções auxiliares:

#Função que realiza a tranformação gamma e mostra o resultado
def power_law(img):
    gamma = [0.3, 0.8, 1.5, 2.0]
    for g in gamma:
        cv.imshow(f'Gamma = {g}', np.array(np.max(img)*(img/np.max(img))**g, dtype = np.uint8))
        cv.waitKey(0)
    cv.destroyAllWindows()

#Função que realiza a equalização de histograma
def equalize(img):
    return cv.equalizeHist(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

#Leitura de todas as imagens
car = cv.imread('car.png')
crowd = cv.imread('crowd.png')
university = cv.imread('university.png')

#Questão 2.1:
#É realizada o realce power-law sobre cada imagem (nome_imagem, valor_gamma)
# e logo após mostrado na tela
power_law(car)
power_law(crowd)
power_law(university)

#Questão 2.2:
#É realiziada a equalização de cada imagem e logo após mostrada na tela
equalized_car = equalize(car)
equalized_crowd = equalize(crowd)
equalized_university = equalize(university)

cv.imshow('equalized_car' , equalized_car)
cv.waitKey(0)

cv.imshow('equalized_crowd' , equalized_crowd)
cv.waitKey(0)

cv.imshow('equalized_university', equalized_university)
cv.waitKey(0)
cv.destroyAllWindows()

#Questão 2.3: Plotar gráficos de histograma e CDF antes e após a equalização da imagem
from matplotlib import pyplot as plt 

#Histograma antes da equalização
carHist = cv.calcHist([car],[0],None,[256],[0,256]) 
plt.subplots(num = 'Histograma original - car.png')
plt.plot(carHist)
plt.show()


#Histograma após a equalização
equalized_carHist = cv.calcHist([equalized_car],[0],None,[256],[0,256]) 
plt.subplots(num = 'Histograma equalizado - car.png')
plt.plot(equalized_carHist)
plt.show()

#CDF antes da equalização
cdf = carHist.cumsum()
plt.subplots(num = 'CDF original - car.png')
plt.plot(cdf)
plt.show()

#CDF após a equalização
cdf = equalized_carHist.cumsum()
plt.subplots(num = 'CDF equalizado - car.png')
plt.plot(cdf)
plt.show()
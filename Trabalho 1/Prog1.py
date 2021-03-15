import cv2 as cv
import numpy as np

from functions import dec_int, egde_improv

img = cv.imread('laranji.jpg')
n = 6

height, width = img.shape[:2]
newWidth = int(width/n)
newHeight = int(height/n)

resized, interpolated = dec_int(img, n)
cubic_resized = cv.resize(img, (newWidth, newHeight), interpolation = cv.INTER_CUBIC)
cubic_interpolated = cv.resize(cubic_resized, (width, height), interpolation = cv.INTER_CUBIC)
interpolated_improv = egde_improv(interpolated)
cubic_interpolated_improv = egde_improv(cubic_interpolated)

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
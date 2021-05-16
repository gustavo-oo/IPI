from functions import *
import cv2 as cv
import numpy as np

images = ["1.jpg", "2.jpg", "3.jpg",  "4.jpg",  "5.jpg",  "6.jpg",  "7.jpg",  "8.jpg",  "9.jpg", "10.jpg", 
            "11.jpg", "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg"]

counter = 0
total = len(images)
print("Processando Imagens ...")
for image in images:
    img = cv.imread(f'./test_images/board_with_pieces/{image}')
    empty_board = cv.imread("./test_images/original.jpg")
    cropped, digitalized = DigitalizeBoard(img, empty_board)

    cv.namedWindow(f'Digitalized - {image}', cv.WINDOW_NORMAL)
    cv.namedWindow(f'Cropped - {image}', cv.WINDOW_NORMAL)
    cv.imshow(f'Digitalized - {image}', digitalized)
    cv.imshow(f'Cropped - {image}', cropped)
    cv.imwrite(f'./result_images/digital_board/{image}', digitalized)
    cv.imwrite(f'./result_images/cropped_board/{image}', cropped)
    cv.waitKey(0)

    counter += 1
    print(f'{int(counter*100/total)}%')
    cv.destroyAllWindows()

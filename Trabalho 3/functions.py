import cv2 as cv
import numpy as np

#Função que preenche os quadrados da matriz 32x32
def FillWhite(img):
    for i in range(0, 32, 4):
        for j in range(0, 32, 4):
            counter = 0
            for x in range(4):
                for y in range(4):
                    if img[i + x][j + y] == 255:
                        counter += 1

            for c in range(4):
                for d in range(4):
                    if counter >= 2:
                        img[i + c][j + d] = 255
                    else:
                        img[i + c][j + d] = 0
    return img

#Função que redimensiona a imagem para 8x8
def Resize(img):
    first_size = (32, 32)
    final_size = (8, 8)

    #Reduzindo para 32x32
    img = cv.resize(img, first_size, interpolation = cv.INTER_LINEAR)
 
    #Preenchendo os pixels 
    img = FillWhite(img)

    #Reduzindo para 8x8
    img = cv.resize(img, final_size, interpolation = cv.INTER_LINEAR)
    return img

#Função que cria o tabuleiro vazio
def CreateChessBoard():
    white_square = (255, 255, 0)
    black_square = (204, 204, 0)
    size = (8, 8, 3)

    digital_board = np.full(size, white_square, dtype = np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            if (i % 2 != 0 and j % 2 == 0) or (i % 2 == 0 and j % 2 != 0):
                digital_board[i][j] = black_square
    return digital_board

#Função que recorta os tabuleiros
def CropImage(img, iterations = 1):
    for i in range(iterations):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #Soma das colunas
        col_sum = np.sum(img_gray, axis = 0)
        row_sum = np.sum(img_gray, axis = 1)

        #Calculo da media
        col_thr = col_sum.sum()/len(col_sum)
        row_thr = row_sum.sum()/len(row_sum)

        #Removendo valores acima da média
        tmp = np.argwhere(row_sum < row_thr)
        x1 = int(tmp[0])
        x2 = int(tmp[-1])

        tmp = np.argwhere(col_sum < col_thr)
        y1 = int(tmp[0])
        y2 = int(tmp[-1])

        #Cortando a imagem
        img = img[x1:x2, y1:y2]

    return img

#Função principal que transforma a imagem em um tabuleiro digital
def DigitalizeBoard(frame, first_frame):

    #Cortando os tabuleiros
    cropped_original = CropImage(first_frame, 5)
    cropped_chessboard = CropImage(frame, 10)

    #Redimensionando para o mesmo tamanho
    size = cropped_original.shape[:2]
    size = size[::-1]
    cropped_chessboard = cv.resize(cropped_chessboard, size, interpolation= cv.INTER_LINEAR)

    cv.imwrite(f"./result_images/digital_board/cropped_original.png", cropped_original)

    #Subtraindo Tabuleiro com peças do tabuleiro sem peças (VAZIO - COM_PEÇAS)
    subtracted_oc = cv.subtract(cropped_original, cropped_chessboard)

    #Subtraindo Tabuleiro sem peças do tabuleiro com peças (COM PEÇAS - VAZIO)
    subtracted_co = cv.subtract(cropped_chessboard, cropped_original)

    #Separando os canais rgb
    blue_oc, green_oc, red_oc = cv.split(subtracted_oc)
    blue_co, green_co, red_co = cv.split(subtracted_co)

    # ======= Identificando Peças pretas em quadrados brancos utilizando o canal azul do subtracted_oc ======= #

    #Binarizando
    BpWs = cv.threshold(blue_oc, 90, 255, cv.THRESH_BINARY)[1]

    #Removendo ruídos
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (40, 40))
    BpWs = cv.morphologyEx(BpWs, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 50))
    BpWs = cv.morphologyEx(BpWs, cv.MORPH_CLOSE, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    BpWs = cv.morphologyEx(BpWs, cv.MORPH_DILATE, kernel)

    #Redimensionando
    BpWs = Resize(BpWs)

    # ====== Identificando Peças brancas em quadrados brancos utilizando o canal vermelho do subtracted_oc ====== #

    #Binarizando
    WpWs = cv.threshold(red_oc, 20, 255, cv.THRESH_BINARY)[1]

    #Removendo Ruídos
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (50, 50))
    WpWs = cv.morphologyEx(WpWs, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 50))
    WpWs = cv.morphologyEx(WpWs, cv.MORPH_CLOSE, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    WpWs = cv.morphologyEx(WpWs, cv.MORPH_DILATE, kernel)

    #Redimensionando
    WpWs = Resize(WpWs)

    #Removendo peças já encontradas
    WpWs = WpWs - BpWs
    
    # ===== Identificando peças brancas nos quadrados pretos utilizando o canal vermelho do subtracted_co ====== #
   
    #Binarização
    WpBs = cv.threshold(red_co, 50, 255, cv.THRESH_BINARY)[1]

    #Removendo Ruídos
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (40, 40))
    WpBs = cv.morphologyEx(WpBs, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20 ))
    WpBs = cv.morphologyEx(WpBs, cv.MORPH_CLOSE, kernel)

    #Redimensionando
    WpBs = Resize(WpBs)
    
    # ====== Identificando peças pretas nos quadrados pretos utilizando o canal verde do subtracted_oc ======= #
    
    #Binarização
    BpBs = cv.threshold(green_oc, 15, 255, cv.THRESH_BINARY)[1]

    #Removendo Ruidos
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (70, 70))
    BpBs = cv.morphologyEx(BpBs, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 30))
    BpBs = cv.morphologyEx(BpBs, cv.MORPH_CLOSE, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    BpBs = cv.morphologyEx(BpBs, cv.MORPH_DILATE, kernel)
  
    #Redimensionado
    BpBs = Resize(BpBs)

    #Removendo Peças já encontradas
    BpBs = BpBs - BpWs - WpBs - WpWs

    # ============= JUNTANDO PEÇAS ENCONTRADAS =================================

    #Criando tabuleiro digital
    digital_board = CreateChessBoard()

   #Adicionando as peças pretas
    Bp = BpWs + BpBs
    for i in range(8):
        for j in range(8):
            if Bp[i][j] == 255:
                digital_board[i][j] = 0

    #Adicionando as peças brancas
    Wp = WpBs + WpWs
    for i in range(8):
        for j in range(8):
            if Wp[i][j] == 255:
                digital_board[i][j] = 255

    #Redimensionando para um tamanho maior para melhor visualização
    digital_board = cv.resize(digital_board, (256, 256), interpolation = cv.INTER_NEAREST)
    return cropped_chessboard, digital_board
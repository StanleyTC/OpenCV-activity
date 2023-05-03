# Comando: pip install opencv-contrib-python
# TEM QUE TER PIP INSTALADO - INSTALAR PYTHON (pesquisar python na microsoft sotre, instalar e o comando
# pip estará habilitado)

# duas opções: escrever import cv2 - vamos usar a função VideoCapture, mas se importarmos toda a biblioteca
# teremos de ficar escrevendo cv2.VideoCapture() toda hora. Para simplificar para apenas escrever 
# VideoCapture() e não ter que escrever cv2 toda hora, vamos para a segunda opção:
# from cv2 import VideoCapture

from cv2 import VideoCapture, createBackgroundSubtractorMOG2, threshold # Essa segunda função será usada na linha 22,
                                                                        # A terceira será para a linha 38 na variavel limiar

# Segundo passo: criar uma variavel para salvar o uso do VideoCapture no arquivo do video:
video = VideoCapture('18_04_23 - 14h16.MOV')   # Isso irá detectar o movimentos do veiculos, com a função VideoCapture, do
                                       # video 18_04_23.MOV, e salvará na variavel video para podermos trabalhar


# Segundo passo: 
# Criar um objeto para aplicar o algoritmo de substração de fundo na imagem atual
# A função para isso é a createBackgroundSubstractorM0G2()

sub = createBackgroundSubtractorMOG2()

# Terceiro passo: aplicar o algoritmo do backgroundsubtractorm0G2 na imagem do video, aplicar segmentação 
# de pixels em movimento, econtrar os contornos dos objetos em movimento e por fim desenhar retângulos ao
# redor dos objetos em movimento (Os retângulos são importantes para podermos ter os nossos movimentos detectados):

import cv2
while True:
    ret, frame = video.read()     # vou criar duas variaveis, ret vai ser um indicador para o frame se ele foi
                                  # lido com sucesso ou não. A variável mais importante, portanto, é o frame, que
                                  # irá armazenar a leitura (com read()) do vídeo sendo processado (o video processado,
                                  # relembrando, está na variável chamada de video)
    if ret == False:
        break

    # algoritmo do backgroundsubstractorm0g2:
    algoritmo = sub.apply(frame)
    # limiar para segmentar pixels em movimento:
    from cv2 import THRESH_BINARY
    limiar = threshold(algoritmo, 200, 255, THRESH_BINARY)[1] # primeiro 300 = X, segundo 300 = y
    # Encontrar contornos dos objetos em movimento:
    from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, contourArea # quarta função será usada na linha 44
    contours, hierarchy = findContours(limiar, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    # Desenhar retangulos ao redor dos objetos em movimento:
    from cv2 import boundingRect, waitKey, destroyAllWindows
    for i in contours:
        area = contourArea(i)
        if area > 1000:
            x, y, w, h = boundingRect(i)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    # Mostrar a imagem com retângulos:
    from cv2 import imshow

    imshow('frame', frame)

    if waitKey(30) == ord('q'):
        break

video.release()
# destroyAllWindows()
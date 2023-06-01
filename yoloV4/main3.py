import cv2
import numpy as np

# Cores
COLORS = [(0, 255, 255),(0, 255, 0),(255, 0, 0)]

# Classes
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# CAPTURA DO VIDEO
cap = cv2.VideoCapture("testinho.mp4")

#CARREGANDO OS PESOS DA REDE NEURAL
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608, 608), scale=1/255)

# LENDO OS FRAMES DO VIDEO
while True:
    
    # CAPUTRA DO FRAME
    _, frame = cap.read()

    # COMEÇO DA CONTAGEM DO MS
    start = cv2.getTickCount()

    # DETECÇÃO
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # FIM DA CONTAGEM DOS MS
    end = cv2.getTickCount()

    # PERCORRER TODAS AS DETECÇÕES
    for (classid, score, box) in zip(classes.flatten(), scores.flatten(), boxes):
        # Converter classid para int
        classid = int(classid)

        #GERANDO UMA COR PARA A CLASSE
        color = COLORS[classid % len(COLORS)]

        # PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
        label = f"{class_names[classid]} : {score}"

        # DESENHANDO A BOX DA DETECÇÃO
        cv2.rectangle(frame, box, color, 2)

        # ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    # CALCULANDO O TEMPO QUE LEVOU PARA FAZER A DETECÇÃO 
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)
    fps_label = f"FPS: {round(fps, 2)}"

    # ESCREVENDO O FPS NA IMAGEM
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # MOSTRANDO A IMAGEM
    cv2.imshow("Detecção de Objetos", frame)

    # ESPERA DA RESPOSTA
    if cv2.waitKey(1) == 27:
        break

# LIBERAÇÃO DA CAMERA E DESTROI TODAS AS JANELAS
cap.release()
cv2.destroyAllWindows()

import cv2
import time
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
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# LENDO OS FRAMES DO VIDEO
while True:
    
    # CAPUTRA DO FRAME
    _, frame = cap.read()

    # COMEÇO DA CONTAGEM DO MS
    start = time.time()

    # DETECÇÃO
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # FIM DA CONTAGEM DOS MS
    end = time.time()
    for i, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
        print(f"Detecção {i+1}:")
        print("classid:", classid)
        print("score:", score)
        print("box:", box)

    # Restante do código...

    # PERCORRER TODAS AS DETECÇÕES
    for (classid, score, box) in zip(classes, scores, boxes):

        #GERANDO UMA COR PARA A CLASSE
        color = COLORS[int(classid) % len(COLORS)]

        # PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
        label = f"{class_names[classid[0]] : {score}}"

        # DESENHANDO A BOX DA DETECÇÃO
        cv2.rectangle(frame, box, color, 2)

        # ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
        cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # CALCULANDO O TEMPO QUE LEVOU PARA FAZER A DETECÇÃO 
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    # ESCREVENDO O FPS NA IMAGEM
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    # MOSTRANDO A IMAGEM
    cv2.imshow("detections", frame)

    # ESPERA DA RESPOSTA
    if cv2.waitKey(1) == 27:
        break

    # LIBERAÇÃO DA CAMERA E DESTROI TODAS AS JANELAS
    cap.release()
    cv2.destroyAllWindows

    
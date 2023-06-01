import cv2
import numpy as np

# Configurações do YOLOv4
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
layer_names = net.getLayerNames()
output_layer_names = net.getUnconnectedOutLayersNames()
output_layers = [layer_names.index(layer) for layer in output_layer_names]

# Carregamento do vídeo
cap = cv2.VideoCapture("testinho.mp4")

# Loop principal
while True:
    # Leitura do próximo frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de objetos usando YOLOv4
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Filtrar detecções com pontuação superior ao limiar
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

    # Desenhar caixas delimitadoras nas detecções
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = class_names[class_ids[i]]
            color = (0, 0, 255)  # Vermelho
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibir o frame resultante
    cv2.imshow("Detecção YOLOv4", frame)
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos e fechar janelas
cap.release()
cv2.destroyAllWindows()

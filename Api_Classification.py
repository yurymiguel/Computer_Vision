import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np


modelo = load_model("keras_model.h5", compile=False)


categorias = ["gato", "cachorro"]


captura = cv2.VideoCapture(0)

while True:

    ret, quadro = captura.read()

    quadro_redimensionado = cv2.resize(quadro, (224, 224))

    quadro_rgb = cv2.cvtColor(quadro_redimensionado, cv2.COLOR_BGR2RGB)

    quadro_entrada = np.expand_dims(quadro_rgb, axis=0)

    previsao = modelo.predict(quadro_entrada)

    categoria = categorias[np.argmax(previsao)]

    cv2.rectangle(quadro, (0, 0),
                  (quadro.shape[1], quadro.shape[0]), (0, 165, 255), 10)

    if categoria == "gato" or categoria == "cachorro":
        cv2.putText(quadro, categoria, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(quadro, 'Não é um Gato nem Cachorro', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Classificador de Cachorro e Gato", quadro)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


captura.release()
cv2.destroyAllWindows()

import matplotlib.pyplot as plt
import requests
import json
import numpy as np
import cv2

CLASSES = ['land_cruser 100', 'qashqai II', 'solyaris', 'vaz 2110', 'x5 e70']
# REST-запрос серверу, расположенныому на вашем компьютере
URL_LOCAL = 'http://localhost:8501/v1/models/img_classifier:predict'
# REST-запрос серверу, расположенныому на виртуальной машине Heroku
URL_HEROKU = 'https://my-car-classification.herokuapp.com/v1/models/model:predict'
# Ссылка на зображение
LINK = './data/test/x5_1.jpg'

def make_prediction(link):
   img = cv2.imread(link)
   img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
   image = np.expand_dims(img, axis=0)
   data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
   headers = {"content-type": "application/json"}
   json_response = requests.post(URL_HEROKU, data=data, headers=headers)
   predictions = json.loads(json_response.text)['predictions']
   return predictions

predictions = make_prediction(LINK)
print(CLASSES[np.argmax(predictions)])






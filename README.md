# car_classification
# Проект по классификации автомобилей

Данный проект разработан для демонстрации возможности нейронных сетей по распознаванию популярных в России моделей автомобилей, а так же споособов интеграции обученной модели. Набор данных состоит из 5 моделей: по 28 изображений для тренировочной выборки и 7 валидационной.
Генерация дасетов изображений осуществляется с помощью ImageDataGenerator библиотеки keras.
Работа состоит из нескольких частей:
1. Часть посвящена предподготовке данных, настройке, обучению и анализу модели (описать выбранную модель и применение Imagedatagenerator) ipynb.
2. Развертывание модели на Docker и тестирование на запущенной модели Heroku.

## 2. Развертывание модели с помощью TensorFlow Serving
[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) — это гибкая высокопроизводительная система обслуживания моделей машинного обучения, разработанная для производственных сред. TensorFlow Serving упрощает развертывание новых алгоритмов и экспериментов, сохраняя при этом ту же архитектуру сервера и API. TensorFlow Serving обеспечивает готовую интеграцию с моделями TensorFlow, но может быть легко расширен для обслуживания других типов моделей. 

Проще говоря, TensorFlow Serving позволяет легко обслуживать модель через сервер моделей. Он предоставляет гибкий API, который можно легко интегрировать в существующую систему или развернуть на облачных платформах. 

### 2.1. Развертывание модели мы будем осуществлять на Ubuntu 20.04 с помощью Docker, а после полученный контейнер интегируем на виртуальную машину Heroku

Порядок установки Docker берем из офциального [руководства]( https://docs.docker.com/engine/install/ubuntu/).
Проверить корректность установки Dockera можно выполнив команду в терминале:

    ~$ docker run hello-world

и в случае правильной установки вы плучите сообщение:

Hello from Docker!
This message shows that your installation appears to be working correctly.

### 2.2. Установка обслуживания Tensorflow
Теперь, когда у вас правильно установлен Docker, вы собираетесь использовать его для загрузки TF Serving. 
В терминале выполните следующую команду:


    ~$  docker pull tensorflow/serving:latest-gpu
 
 а если вы хотите использовать CPU то: 
 
    ~$ docker pull tensorflow/serving:latest-gpu
 
 и в случае правильной установки вы плучите сообщение:

    Status: Image is up to date for tensorflow/serving:latest-gpu
docker.io/tensorflow/serving:latest-gpu

### 2.3. Обслуживание сохраненной модели с помощью Tensorflow Serving
Подробный порядок развертывания модели описан в [руководстве](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
Папка с сохраненной моделью должна быть представлена в следющем виде:

├── img_classifier 

│ ├── 1600788643 

│ │ ├── assets 

│ │ ├── save_model.pb 

│ │ └── переменные

После сохранения модели и правильной установки Tensorflow Serving с Docker, вы будете использовать ее в качестве конечной точки API.
Tensorflow Serving допускает два типа конечных точек API — REST и gRPC:

REST — это коммуникационный «протокол», используемый веб-приложениями. Он определяет стиль общения клиентов с веб-сервисами. Клиенты REST взаимодействуют с сервером, используя стандартные методы HTTP, такие как GET, POST, DELETE и т. д. Полезная нагрузка запросов в основном кодируется в формате JSON.
С другой стороны, gRPC — это протокол связи, изначально разработанный в Google. Стандартный формат данных, используемый с gRPC, называется буфером протокола . gRPC обеспечивает связь с малой задержкой и меньшую полезную нагрузку, чем REST, и предпочтительнее при работе с очень большими файлами во время логического вывода. 

В папке проекта откройте терминал и добавьте команду Docker ниже:

    docker run -p 8501:8501 --name tf_car_classifier \
                        --mount type=bind,source=/home/cooper/my_projects/my-car-classification/img_classifier,\
                        target=/models/img_classifier \
                        -e MODEL_NAME=img_classifier -t tensorflow/serving
                        
 где 
 
`-p 8501:8501`: это порт конечной точки REST. Каждый запрос прогнозирования будет направляться на этот порт. Например, вы можете сделать запрос прогноза на http://localhost:8501 .
 
`-- name tf_car_classifier`: это имя, данное контейнеру Docker, на котором работает TF Serving. Его можно использовать для запуска и остановки экземпляра контейнера позже. 

`-- mount type=bind,source=/Users/tf-server/img_classifier/,target=/models/img_classifier`: Команда mount просто копирует модель по указанному абсолютному пути ( /home/cooper/my_projects/my-car-classification/img_classifier ) в Docker контейнер ( /models/img_classifier ), чтобы у TF Serving был к нему доступ. 

`-e MODEL_NAME=img_classifier`: имя модели  для запуска. Это имя, которое вы использовали для сохранения вашей моделипри внутри контейнера.

`-t tensorflow/serving`: Контейнер TF Serving Docker для запуска.

Выполнение приведенной выше команды запускает контейнер Docker, а TF Serving предоставляет конечные точки gRPC (0.0.0.0:8500) и REST (localhost:8501).
Вставить фото...

### 2.4.  Создание запроса к модели  помощью протокала REST

Теперь, когда конечная точка запущена и работает, мы можем сделать к ней вызовы логического вывода через HTTP-запрос. Реализация предобработки изображения, создание запроса и получение ответа от  сервера реализована в predict.py, в папке проекта откройте терминал и запустите файл.

### 2.5.  Развертывание контейнера на Heroku

https://doc.cuba-platform.com/manual-latest-ru/heroku_container.html


Подробный порядок развертывания модели описан в [руководстве](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)






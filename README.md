# car_classification
***Проект активно дорабатывается, и будет закончен в ближайшее время***
# Проект по классификации автомобилей

Данный проект разработан для демонстрации возможности классификации изображаний с применением нейронных сетей. 
Набор данных состоит из  пяти  моделей автомобилей: solyaris, land cruiser, bmw x5, vaz 2110 и qashqai, которые расположены в папке data и состоят из тренировочных и валидационных наборов по 5 класов каждая, соотношение 28 изображений для тренировки и 9 для валидации, а так же папка test в которую можно добавлять любые изображения представленных марок автомобилей.
Я попытался реализовать полный жизненный цикл модели, который можно представить в следующем виде :
![Снимок экрана от 2022-04-03 20-21-45](https://user-images.githubusercontent.com/80042896/161440431-4c64d323-2567-4c00-9f32-fb68ecb31c75.png)

- первая часть посвящена предобработке данных, выбору, настройке и обучению модели с последующей проверкой работы и  анализом полученных результатов. Все вышеперечисленное реализуется в ноутбуке [training_on_imagedatagenerator_5_classes.ipynb](https://github.com/manzhura/car_classification/blob/main/training_on_imagedatagenerator_5_classes.ipynb).
- вторая часть посвещена развертыванию модели (deployment) с помощью API TensorFlow Serving на локальном и удаленном сервере, который реализован на Heroku.

Так же хочется обратить внимание, что в данной работе не раскрывается вся сила tensorflow в том числе построение эффективных конвееров, выбор каких-то супер сложных/соверменных  моделей и т.д., это связано с тем, что набор данных достаточно мал и этого не требуется. В данном проекте продемонстрированы базовые принципы построения датасетов и обучения моделей, а так же их деплоинг. Весь проект реализован на операционной системе Ubuntu 20.04, перечень необходимых для работы библиотек и фреймворков представлен в файле [requirements.txt](https://github.com/manzhura/car_classification/blob/b3e13217d25ef1e498454baa227c39698bda984d/requirements.txt).

## 2. Развертывание модели с помощью TensorFlow Serving
[***TensorFlow Serving***](https://www.tensorflow.org/tfx/guide/serving) — это гибкая высокопроизводительная система обслуживания моделей машинного обучения, разработанная для производственных сред. TensorFlow Serving обеспечивает готовую интеграцию с моделями TensorFlow, но может быть легко расширен для обслуживания других типов моделей. Проще говоря, TensorFlow Serving позволяет легко обслуживать модель через сервер моделей, он предоставляет гибкий API, который можно легко интегрировать в существующую систему или развернуть на облачных платформах.
 ![tensorflow-serving](https://user-images.githubusercontent.com/80042896/161436971-a85dfe2e-5ce1-4c98-9d77-27973ca90fb8.png)

<!-- Развертывание модели мы будем осуществлять на Ubuntu 20.04 с помощью Docker, а после полученный контейнер интегируем на виртуальную машину Heroku -->

### 2.1. Установка Docker 

Один и самых простых способов способов, чтобы начать использовать TensorFlow Serving это установка и работа с Docker. 
Порядок установки Docker  представлен в [руководстве]( https://docs.docker.com/engine/install/ubuntu/).

Проверить корректность установки Docker можно, выполнив команду в терминале:

    ~$ docker run hello-world

и в случае правильной установки вы получите сообщение:

    Hello from Docker!
    This message shows that your installation appears to be working correctly.

### 2.2. Установка обслуживания Tensorflow

Теперь, когда у нас правильно установлен Docker, мы можем использовать его для загрузки TF Serving. 
В терминале выполним следующую команду:

    ~$  docker pull tensorflow/serving:latest-gpu
 
 а если хотим использовать только CPU то: 
 
    ~$ docker pull tensorflow/serving:
 
 и в случае правильной установки получим сообщение:

    Status: Image is up to date for tensorflow/serving:latest-gpu
    docker.io/tensorflow/serving:latest-gpu

### 2.3. Обслуживание сохраненной модели с помощью Tensorflow Serving
Подробный порядок развертывания модели описан в [руководстве](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple).
В первой части проекта мы сохранили обученную модель model_unfreez в папку img_classifier/1, папка с моедлью должна быть представлена в следующем виде:

├── models

│ ├── 1 

│ │ ├── assets 

│ │ ├── save_model.pb 

│ │ └── переменные

После сохранения модели и правильной установки Tensorflow Serving с Docker, мы будем использовать ее в качестве конечной точки API.
Tensorflow Serving допускает два типа конечных точек API — *REST* и *gRPC*:

***REST*** — это коммуникационный «протокол», используемый веб-приложениями. Он определяет стиль общения клиентов с веб-сервисами. Клиенты REST взаимодействуют с сервером, используя стандартные методы HTTP, такие как GET, POST, DELETE и т. д. Полезная нагрузка запросов в основном кодируется в формате JSON.

***gRPC*** — это протокол связи, изначально разработанный в Google. Стандартный формат данных, используемый с gRPC, называется буфером протокола. 
gRPC обеспечивает связь с малой задержкой и меньшую полезную нагрузку, чем REST, и предпочтительнее при работе с очень большими файлами во время логического вывода. 

В корневой папке проекта откройте терминал и добавьте команду Docker ниже:

      ~$  docker run -p 8501:8501 --name tf_car_classifier \
                        --mount type=bind,source=/home/cooper/my_projects/my-car-classification/models
                        target=/models/img_classifier \
                        -e MODEL_NAME=img_classifier -t tensorflow/serving
                        
 где 
 
`-p 8501:8501`: это порт конечной точки REST. Каждый запрос прогнозирования будет направляться на этот порт. Например, вы можете сделать запрос прогноза на http://localhost:8501 .
 
`--name tf_car_classifier`: это имя, данное контейнеру Docker, на котором работает TF Serving. Его можно использовать для запуска и остановки экземпляра контейнера позже. 

`--mount type=bind,source= /home/cooper/my_projects/my-car-classification/models`: Команда mount просто копирует модель по указанному абсолютному пути (`/home/cooper/my_projects/my-car-classification/models`) в Docker контейнер ( /models/img_classifier ), чтобы у TF Serving был к нему доступ. 

`-e MODEL_NAME=img_classifier`: имя модели  для запуска. Это имя, которое вы использовали для сохранения вашей моделипри внутри контейнера.

`-t tensorflow/serving`: Контейнер TF Serving Docker для запуска.

Выполнение приведенной выше команды запускает контейнер Docker, а TF Serving предоставляет конечные точки gRPC (0.0.0.0:8500) и REST (localhost:8501).
Проверить созданный образ контейнера можно выполнив команду:
 
 `~$  docker ps -a`

Базовые команды для управления контейнером, которые могут пригодится на этом этапе: stop, restart, rm, ps

![image]
(https://user-images.githubusercontent.com/80042896/160816575-d33fe7e8-1564-434a-b14b-39d9a0e5c888.png)


### 2.4.  Создание запроса к модели  помощью протокала REST

Теперь, когда конечная точка запущена и работает, мы можем сделать к ней вызовы  через HTTP-запрос. Запрос к серверу реализован в файле predict.py, Вам будет необходимо корреткно заполнить URL_LOCAL и LINK c учетом размещения данных и модели на Вашем компьютере.
в папке проекта откройте терминал и запустите файл.

~$ python3 predict.py

### 2.5.  Развертывание контейнера на Heroku.

Для начала небходимо зарегестрироваться на Heroku (https://doc.cuba-platform.com/manual-latest-ru/heroku_war_deployment.html), если у вас нет аккаунта.
Всю работу по созданию удаленного сервера мы будем проводить из терминада, открытого внутри проекта. 

Для входа в аккаунт введите:

  ~$  heroku login

Полезно убедиться, что у вас создано не более 5 проектов, это важно для бесплатного использования Heroku:)

 ~$  sudo heroku apps

и создать новый проект, вмоем случае это "my-car-classification"

~$  heroku create my-car-classification

При развертывании  контейнера на Heroku,  столкнемся с последней небольшой трудностью, оОбслуживание Tensorflow обслуживает Rest API через порт 8501, но Heroku назначает случайный порт при запуске dyno. Следовательно, необходимо обновить порт по умолчанию для команды tf-serving, решения данной проблемы реализовано в файле Docker.
 
~$  heroku container:push web -a my-car-classification


Запрос к серверу реализован аналогично п.п. 2.4, за исключением URL адреса вашей модели, развернутой на удаленном сервере  файле.
Если вы не хотите заморачиваться, то можете просто сделать запрос на мой сервер, который пока работает в режиме онлан


https://doc.cuba-platform.com/manual-latest-ru/heroku_container.html


Подробный порядок развертывания модели описан в [руководстве](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple).






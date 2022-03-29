# car_classification
# Проект по классификации автомобилей

Данный проект разработан для демонстрации возможности нейронных сетей по распознаванию популярных в России моделей автомобилей, а так же споособов интеграции обученной модели. Набор данных состоит из 5 моделей: по 28 изображений для тренировочной выборки и 7 валидационной.
Генерация дасетов изображений осуществляется с помощью ImageDataGenerator библиотеки keras.
Работа состоит из нескольких частей:
1. Часть посвещена предподготовке данных, настройке, обучению и анализу модели (описать выбранную модель и применение Imagedatagenerator) ipynd.
2.  Развертывание модели на Docker и тестирование на запущенной модели heroku.

#  Развертывание модели с помощью tensorflow server
Подробный порядок развертывания модели описан в https://www.tensorflow.org/tfx/tutorials/serving/rest_simple, а описание краткое взять от седа -https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker
[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) — это гибкая высокопроизводительная система обслуживания моделей машинного обучения, разработанная для производственных сред. TensorFlow Serving упрощает развертывание новых алгоритмов и экспериментов, сохраняя при этом ту же архитектуру сервера и API. TensorFlow Serving обеспечивает готовую интеграцию с моделями TensorFlow, но может быть легко расширен для обслуживания других типов моделей. 

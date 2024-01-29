# **_Face Recognizer_**
The face recognition project.

## ✏️ Описание проекта
Проект предназначен показать работу, в котором программа _тренируется запоминать лица людей с помощью фотографий_ и \
в дальнейшем _распознавать этих людей по другим фотографиям_. (За основу были взяты разные знаменитости)

### 📋 Задание
Найти лицо на фотографии и распознать.

### 🛠️ Команды для работы с проектом
- `python detector.py -h, --help` - показать все команды.
- `python detector.py --train` - тренировка на входных данных.
- `python detector.py --validate` - проверка обученной модели.
- `python detector.py --test` - протестировать модель с неизвестным изображением.
- `python detector.py --train -m="hog"` - использовать модель _hog_ (CPU) для обучения:
- `python detector.py --train -m="cnn"` - использовать модель _cnn_ (GPU) для обучения:
- `python detector.py --test -f unknown.jpg` - протестировать модель с неизвестным лицом передав _путь к изображению_.

## 📽️ Пример работы проекта
![pycharm64_ADMyTwqgwc.gif](..%2F..%2F..%2FUsers%2FAkuev%2FDocuments%2FShareX%2FScreenshots%2F2024-01%2Fpycharm64_ADMyTwqgwc.gif)
from pathlib import Path

from face_work.commands import args
from face_work.face_encode import FaceEncode
from face_work.face_recognizer import FaceRecognizer

# Если этих каталогов нет, то они создадутся.
Path('training').mkdir(exist_ok=True)
Path('output').mkdir(exist_ok=True)
Path('validation').mkdir(exist_ok=True)

if __name__ == '__main__':
    # Тренировка на входных данных.
    if args.train:
        FaceEncode.encode_known_faces(model=args.m)
    # Проверка обученной модели.
    if args.validate:
        FaceRecognizer.validate(model=args.m)
    # Тест модели с неизвестным изображением.
    if args.test:
        FaceRecognizer.recognize_faces(image_location=args.f, model=args.m)

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import face_recognition

from face_work.face import Face

if TYPE_CHECKING:
    from numpy import ndarray


class FaceEncode(Face):
    """Класс кодирование лиц."""
    @staticmethod
    def __encode_known_face(model: str) -> tuple[list[str], list[list[ndarray]]]:
        """Проходимся по каждому известному лицу и кодируем его."""
        names = []
        encodings = []

        for filepath in Path('training').glob('*/*'):
            # Сохранение метку каждого каталога.
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)

            # Определяет местоположение лиц на каждом изображении и возвращает
            # список кортежей, в которых расположены
            # четыре элемента (координаты лица).
            face_locations = face_recognition.face_locations(image, model=model)

            # Используется для генерации кодировок (числовое представление черт лица)
            # для обнаруженных лиц на изображении.
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # Добавление `имен` и их `кодировок` в отдельные списки.
            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)
        return names, encodings

    @classmethod
    def encode_known_faces(
            cls,
            # Метод обнаружения объектов, так же есть `cnn`.
            model: str = 'hog',
    ) -> None:
        """Закодировать известные лица."""
        names, encodings = cls.__encode_known_face(model=model)
        name_encodings = {'names': names, 'encodings': encodings}
        cls._write_or_read_file_binary_form(mode='wb', name_encodings=name_encodings)

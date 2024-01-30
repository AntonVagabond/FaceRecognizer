from __future__ import annotations

import collections
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import face_recognition
from PIL import Image, ImageDraw

from face_work.face import Face

if TYPE_CHECKING:
    from numpy import ndarray
    from PIL.ImageDraw import ImageDraw


class FaceRecognizer(Face):
    """Класс распознавания лиц."""
    @staticmethod
    def __recognize_face(
            unknown_encoding: ndarray,
            loaded_encodings: dict[str, Union[list[list[ndarray]], list[str]]]
    ) -> Optional[str]:
        """Распознавание лица."""
        # Сравниваем каждую неизвестную кодировку с
        # кодировками, которые загрузили раннее.
        boolean_matches = face_recognition.compare_faces(
            loaded_encodings['encodings'], unknown_encoding
        )

        # Отслеживаем сколько голосов имеет каждое потенциальное совпадение
        # и возвращаем имя за которое больше всего набрано голосов.
        votes = collections.Counter(
            name
            for match, name in zip(boolean_matches, loaded_encodings['names'])
            if match
        )
        return votes.most_common(1)[0][0] if votes else None

    @classmethod
    def __display_face(
            cls,
            draw: ImageDraw,
            bounding_box: tuple[int, int, int, int],
            name: str,
    ) -> None:
        """
        Отобразить лицо.
        Метод рисует ограничивающий прямоугольник на распознанном лице и добавляет к
        этому ограничивающему прямоугольнику подпись, если лицо идентифицировано.
        """
        # Координаты для отрисовки прямоугольника вокруг распознанного лица.
        top, right, bottom, left = bounding_box

        # Отрисовка прямоугольника вокруг распознанного лица.
        draw.rectangle(
            xy=((left, top), (right, bottom)),
            outline=cls.BOUNDING_BOX_COLOR,
        )

        # Определяем ограничивающею рамку для текстовой подписи.
        text_left, text_top, text_right, text_bottom = draw.textbbox(
            xy=(left, bottom), text=name,
        )

        # Отрисовка ограничивающего прямоугольника вокруг текстовой подписи.
        draw.rectangle(
            xy=((text_left, text_top), (text_right, text_bottom)),
            fill=cls.BOUNDING_BOX_COLOR,
            outline=cls.BOUNDING_BOX_COLOR,
        )

        # Отображение имени в поле подписи.
        draw.text(xy=(text_left, text_top), text=name, fill=cls.TEXT_COLOR)

    @classmethod
    def recognize_faces(cls, image_location: str, model: str = 'hog') -> None:
        """В этот метод попадают немаркированные изображения."""
        loaded_encodings = cls._write_or_read_file_binary_form(mode='rb')

        # Загрузить изображение, на котором нужно распознать лица.
        input_image = face_recognition.load_image_file(image_location)

        # Определить координаты лица на изображении.
        input_face_locations = face_recognition.face_locations(
            input_image, model=model
        )

        # Получить кодировку для обнаруженного лица.
        input_face_encodings = face_recognition.face_encodings(
            input_image, input_face_locations
        )

        # Создать объект Pillow image из загруженного входного изображения.
        pillow_image = Image.fromarray(input_image)

        # Создать объект рисования изображения, который поможет вам нарисовать
        # ограничивающую рамку вокруг граней.
        draw = ImageDraw.Draw(pillow_image)

        for bounding_box, unknown_encodings in zip(
                input_face_locations, input_face_encodings
        ):
            name = cls.__recognize_face(unknown_encodings, loaded_encodings)
            name = 'Unknown' if name is None else name
            cls.__display_face(draw, bounding_box, name)

        # Вручную удаляем объект рисования из текущей области.
        del draw
        # Показываем изображение.
        pillow_image.show()

    @classmethod
    def validate(cls, model: str = 'hog') -> None:
        """Проверка модели."""
        # Открываем каталог проверки (Path), а затем получаем
        # все файлы в этом каталоге (.rglob).
        for filepath in Path('validation').rglob('*'):
            # Подтверждаем, что ресурс является файлом.
            if filepath.is_file():
                cls.recognize_faces(
                    image_location=str(filepath.absolute()), model=model
                )

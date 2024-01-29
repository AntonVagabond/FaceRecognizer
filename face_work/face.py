from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from numpy import ndarray


class Face:
    """Базовый класс лица."""
    # Путь кодирования по умолчанию
    DEFAULT_ENCODING_PATH = Path('output/encodings.pkl')
    BOUNDING_BOX_COLOR = 'blue'
    TEXT_COLOR = 'white'

    @classmethod
    def _write_or_read_file_binary_form(
            cls,
            mode: str,
            name_encodings: Optional[
                dict[str, Union[list[str]], list[list[ndarray]]]
            ] = None,
    ) -> Optional[dict[str, Union[list[str]], list[list[ndarray]]]]:
        """
        Записать закодированные лица в файл, в бинарном виде.
        Либо
        Преобразовать файл из бинарного вида в нормальный и прочитать его.
        """
        if mode == 'rb':
            with cls.DEFAULT_ENCODING_PATH.open(mode='rb') as f:
                # Открыть и загрузить сохраненные кодировки лиц.
                return pickle.load(f)
        with cls.DEFAULT_ENCODING_PATH.open(mode='wb') as f:
            pickle.dump(name_encodings, f)

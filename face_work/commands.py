import argparse

parser = argparse.ArgumentParser(description='Распознавание лиц на изображении')
# Запустит процесс обучения. Вы можете дополнительно указать, использовать ли
# метод `HOG` на базе CPU или `CNN` на базе GPU.
parser.add_argument(
    '--train', action='store_true', help='Тренировка на входных данных',
)
# Запустит процесс проверки, в ходе которого модель делает снимки с известными
# лицами и пытается правильно идентифицировать их.
parser.add_argument(
    '--validate', action='store_true', help='Проверка обученной модели',
)
# Это опция, которую вы, вероятно, будете использовать чаще всего.
# Используйте ее вместе с опцией -f, чтобы указать местоположение изображения с
# неизвестными лицами, которые вы хотите идентифицировать.
# По сути, это работает так же, как проверка, за исключением того, что вы сами
# указываете местоположение изображения.
parser.add_argument(
    '--test',
    action='store_true',
    help='Протестируйте модель с неизвестным изображением',
)
parser.add_argument(
    '-m',
    action='store',
    default='hog',
    choices=['hog', 'cnn'],
    help='Какую модель использовать для обучения: hog (CPU), cnn (GPU)',
)
# Указать местоположение изображения с неизвестными лицами,
# которые вы хотите идентифицировать.
parser.add_argument(
    '-f', action='store', help='Путь к изображению с неизвестным лицом',
)
args = parser.parse_args()
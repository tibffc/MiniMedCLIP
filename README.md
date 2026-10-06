# MiniCLIP / Mini-MedCLIP experiments


## Структура проекта

```text
clip_medclip/
├── clip_medclip.ipynb   # основной ноутбук с экспериментами
├── README.md
├── clip_medclip/        # Python-пакет с логикой проекта
│   ├── __init__.py
│   ├── config.py                   # пути, размеры, seeds и параметры экспериментов
│   ├── data.py                     # загрузка и подготовка данных
│   ├── text.py                     # токенизация и текстовые prompt'ы
│   ├── prompts.py                  # медицинские prompt-наборы
│   ├── miniclip.py                 # MiniCLIP: модель и обучение
│   ├── medclip.py                  # Mini-MedCLIP и semantic loss
│   ├── evaluation.py               # linear probe и метрики
│   └── utils.py                    # device, seed и вспомогательные функции
├── data/                           # данные (не входят в архив)
├── checkpoints/                    # сохранённые модели
├── cache/                          # кэш подготовленных данных
└── results/
    └── figures/                    # графики экспериментов
```

## Эксперименты

Ноутбук содержит только экспериментальный слой:

- настройку/выбор эксперимента;
- вызов функций подготовки данных и моделей;
- запуск обучения;
- zero-shot / linear-probe эксперименты;
- сравнение моделей и результатов;
- построение графиков и таблиц.

Архитектура моделей, загрузка данных, prompt'ы, loss, evaluation и прочая переиспользуемая логика находятся в `.py`-модулях.

## Данные

Проект ожидает датасет COVID-19 Radiography Dataset в следующем месте:

```text
data/
└── COVID-19_Radiography_Dataset/
```

Путь задаётся в `clip_medclip/config.py` через `DATA_ROOT`.

> Датасет не включён в архив проекта.

Для части экспериментов также используется CIFAR-10. При первом запуске соответствующий датасет может быть загружен через `torchvision`.

## Установка

Рекомендуется Python 3.10+ и отдельное виртуальное окружение.

Пример установки основных зависимостей:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\activate       # Windows

pip install torch torchvision numpy pandas scikit-learn pillow tqdm matplotlib jupyter
```

## Параметры экспериментов

Основные параметры находятся в:

```text
clip_medclip/config.py
```

Там задаются, в частности:

- `IMG_SIZE` — размер изображения;
- `EMBED_DIM` — размер embedding;
- `LP_TRAIN_N` — размер обучающей выборки для linear probe;
- `MEDCLIP_EPOCHS` — число эпох Mini-MedCLIP;
- `SEEDS` — набор random seeds;
- `DATA_ROOT`, `RESULTS_DIR`, `CKPT_DIR`, `CACHE_DIR` — пути проекта.

## Результаты

Во время работы создаются следующие каталоги:

```text
checkpoints/   # веса моделей
cache/         # промежуточные/кэшированные данные
results/       # результаты экспериментов
results/figures/  # сохранённые графики
```

Эти каталоги создаются автоматически при импорте `config.py`.

## Быстрый workflow

Типичный сценарий работы выглядит так:

1. Положить `COVID-19_Radiography_Dataset` в `data/`.
2. Открыть `clip_medclip.ipynb`.
3. Настроить параметры эксперимента в `config.py` при необходимости.
4. Запустить подготовку данных.
5. Обучить/загрузить MiniCLIP и Mini-MedCLIP.
6. Выполнить zero-shot и linear-probe evaluation.
7. Сравнить результаты и построить графики непосредственно в ноутбуке.

## Ссылки

1. arXiv:2103.00020

Рэдфорд А., Ким Дж. В., Халласи К. и др. Learning Transferable Visual Models From Natural Language Supervision [Электронный ресурс]. — arXiv:2103.00020, 2021. — Режим доступа: https://arxiv.org/abs/2103.00020 .

2. arXiv:2210.10163

Эслами С., де Мело Г., Мейнель К. MedCLIP: Contrastive Learning from Unpaired Medical Images and Text [Электронный ресурс]. — arXiv:2210.10163, 2022. — Режим доступа: https://arxiv.org/abs/2210.10163.

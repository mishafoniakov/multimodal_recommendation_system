# Multimodal Recommendation System

Дипломный проект: **рекомендация похожих изображений** по мультимодальным признакам.

Система отвечает на вопрос: *к этой картинке какие ещё картинки относятся?*  
Для этого у каждого изображения есть два представления:

- **CLIP** — вектор самой картинки (`openai/clip-vit-base-patch32`, 512 чисел);
- **BERT** — вектор текста про эту картинку (`bert-base-multilingual-cased`, 768 чисел, берётся токен `[CLS]`).

Дальше картинки кластеризуются в общем CLIP+BERT пространстве, из ближайших кластеров собираются пары, и классификатор учится отличать **связанную пару** от случайной. На инференсе для запроса берутся топ-5 кандидатов трёх способов: модель, CLIP cosine, BERT cosine.

Два источника данных:

| Источник | `--source` | Откуда картинки | Откуда текст |
|---|---|---|---|
| Яндекс (пары related images) | `yandex` | скачивание URL из TSV | подписи со страниц / reverse image search |
| Flickr30k | `flickr` | папка `flickr30k_images` | 5 английских подписей на кадр |

---

## Как устроен пайплайн

```
картинки + тексты
        │
        ▼
 CLIP-векторы          BERT-векторы
        │                    │
        └──────── join ──────┘
                    │
                    ▼
        KMeans по CLIP∥BERT
                    │
                    ▼
     пары (image, candidate)
     + cosine / L2 по CLIP и BERT
                    │
                    ▼
     обучение классификатора
                    │
                    ▼
     топ-5: модель / CLIP / BERT
                    │
                    ▼
              коллаж для оценки
```

Каждый прямоугольник — отдельная папка `stage_XX_*` с классами и своим `run.py`.

---

## Структура репозитория

```
multimodal_recommendation_system/
├── run.py                          # общая точка входа
├── requirements.txt
├── .env.example                    # шаблон путей и гиперпараметров
├── config/
│   └── settings.py                 # Settings: пути, модели, k, top-n
├── common/
│   ├── io.py                       # JsonlStore — JSON Lines
│   ├── metrics.py                  # cosine и L2 через numpy
│   └── logging_setup.py
├── stage_01_collection/            # сбор картинок и текстов
│   ├── image_collector.py          # ImageCollector
│   ├── text_scraper.py             # YandexImageScraper
│   └── text_aggregator.py          # TextAggregator
├── stage_02_clip/                  # CLIP
│   └── clip_embedder.py            # ClipEmbedder
├── stage_03_bert/                  # BERT
│   ├── bert_embedder.py            # BertEmbedder
│   └── merger.py                   # EmbeddingMerger
├── stage_04_clustering/            # кластеры и пары
│   └── clusterer.py                # ClusterPipeline
├── stage_05_training/              # обучение
│   ├── dataset.py                  # LearningDataset
│   └── trainers.py                 # CatBoost / XGB / LGBM / DT / RF
├── stage_06_candidates/            # топ-5 кандидатов
│   └── ranker.py                   # CandidateRanker
├── stage_07_evaluation/            # визуальное сравнение
│   └── visualizer.py               # ResultVisualizer
└── data/                           # локальные датасеты (не в git)
```

Результаты запуска пишутся в `artifacts/<source>/`. Эта папка в git не входит.

---

## Установка

Нужны Python 3.10+ и (по возможности) GPU для CLIP/BERT.

```bash
git clone https://github.com/mishafoniakov/multimodal_recommendation_system.git
cd multimodal_recommendation_system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` укажите, где лежат картинки и уже посчитанные векторы.  
Готовые файлы диплома (`clip_bert.json`, `learning_pipeline.csv`) хранятся локально и **не копируются в git** — они по ~3 ГБ каждый:

```bash
THESIS_DATA_DIR=/path/to/local/datasets
YANDEX_IMAGES_DIR=/path/to/yandex/images
FLICKR_IMAGES_DIR=/path/to/flickr30k_images
FLICKR_CAPTIONS_CSV=/path/to/flickr_captions.csv
```

Прокси для сбора текстов Яндекса — только `PROXY_URL`, не в исходниках.

---

## Запуск

Общая команда:

```bash
python run.py --stage <этап> --source yandex|flickr [--model catboost]
```

| `--stage` | Что делает |
|---|---|
| `collection` | собирает тексты (Flickr CSV или `result_*.json` Яндекса) |
| `clip` | считает CLIP по папке картинок |
| `bert` | считает BERT и склеивает с CLIP по имени файла |
| `cluster` | MiniBatchKMeans, пары из ближайших кластеров, cosine/L2 |
| `train` | обучение модели (`--model catboost\|xgboost\|lgbm\|dt\|rf`) |
| `candidates` | топ-5 кандидатов модели, CLIP и BERT |
| `eval` | коллаж запроса и 5+5+5 кандидатов |
| `all` | этапы подряд (сбор картинок по TSV сюда не входит) |

Примеры:

```bash
python run.py --stage clip --source yandex
python run.py --stage train --source yandex --model catboost
python run.py --stage candidates --source flickr
python run.py --stage eval --source yandex
```

Тот же этап можно запустить из своей папки:

```bash
python stage_04_clustering/run.py --source yandex
python stage_05_training/run.py --source yandex --model rf
python stage_07_evaluation/run.py --source flickr --image 10002456.jpg
```

Скачивание пар Яндекса (нужен TSV `related_images.tsv`):

```bash
python stage_01_collection/run.py --source yandex --part images --tsv /путь/related_images.tsv --start 0 --end 1000
```

---

## Этапы подробно

### 1. Сбор данных — `stage_01_collection`

**`ImageCollector`** читает TSV связанных картинок Яндекса, качает JPEG, помнит URL→файл в `dataset_dict.txt`, пишет `dataset.json` (одна строка — одна пара).

**`YandexImageScraper`** ходит в поиск Яндекса по картинке и забирает подпись плюс соседние URL. Пауза между запросами по умолчанию 15 секунд.

**`TextAggregator`**

- Яндекс: склеивает куски текста из `result_*.json` в `{img: txt}`;
- Flickr: читает CSV с разделителем `|`, чистит подписи и объединяет пять комментариев к одному кадру.

Выход: `image_big_text_dataset.json` (JSON Lines: `img`, `txt`).

### 2. CLIP — `stage_02_clip`

**`ClipEmbedder`** открывает картинки батчами, прогоняет CLIP и пишет `clip.json` / `clip_flickr.json`:

```json
{"img": "img_12_0.jpg", "clip": [0.01, -0.22, ...]}
```

Размер вектора — 512.

### 3. BERT и склейка — `stage_03_bert`

**`BertEmbedder`** токенизирует текст (`max_length=50`, как в дипломе) и берёт вектор `[CLS]` (768 чисел).

**`EmbeddingMerger`** делает inner join CLIP и BERT **по имени файла**, без двойного цикла. Выход — `clip_bert.json`:

```json
{"img": "img_12_0.jpg", "clip": [...512...], "bert": [...768...]}
```

### 4. Кластеры — `stage_04_clustering`

**`ClusterPipeline`**

1. Склеивает CLIP∥BERT в один вектор длины 1280.
2. `MiniBatchKMeans` (по умолчанию 1000 кластеров).
3. Для каждой картинки берёт `NEAREST_CLUSTERS_NUMBER` ближайших центров (по умолчанию 3).
4. Все картинки этих кластеров становятся кандидатами, кроме самой себя.
5. Для каждой пары считает четыре числа: `clip_cossim`, `clip_l2`, `bert_cossim`, `bert_l2` (numpy, не цикл Python).

Выход: `nearest_clusters.json`, таблица пар, `pairs_with_metrics.csv`.

Зачем кластеры: полный перебор пар неподъёмен. Кандидаты ищутся только среди «соседних» кластеров.

### 5. Обучение — `stage_05_training`

**`LearningDataset`**

- читает `learning_pipeline.csv`;
- делит 60 / 20 / 20 на train / val / test;
- масштабирует четыре метрики близости **только по train** (без утечки в test);
- для sklearn/XGBoost/LightGBM раскладывает CLIP/BERT-списки в колонки.

**Модели** (общий интерфейс `ModelTrainer`: `fit` → `evaluate` → `save`):

| `--model` | Класс | Заметка |
|---|---|---|
| `catboost` | `CatBoostTrainer` | основная модель диплома, `Logloss`, `scale_pos_weight=2.64`, early stopping |
| `xgboost` | `XGBoostTrainer` | 2000 деревьев, lr 0.01, depth 6 |
| `lgbm` | `LightGBMTrainer` | те же порядки гиперпараметров |
| `dt` | `DecisionTreeTrainer` | sklearn |
| `rf` | `RandomForestTrainer` | sklearn |

CatBoost дополнительно получает `group_id=query` (номер исходной картинки), если колонка `query` есть в таблице.

Метрики на тесте: accuracy и ROC-AUC, график `artifacts/<source>/plots/roc_<model>.png`.  
Модель: `artifacts/<source>/<model>_model.bin`.

### 6. Кандидаты — `stage_06_candidates`

**`CandidateRanker`**

- масштабирует 4 метрики;
- если есть `catboost_model.bin`, считает `P(пара связана)`;
- для каждого запроса берёт 5 лучших по `pred_1`, по CLIP cosine и по BERT cosine.

Выход (JSON Lines `img` + `candidates`):

- Яндекс: `image_model_candidates_5.json`, `image_candidates_5_clip_cossim.json`, `image_candidates_5_bert_cossim.json`
- Flickr: те же смыслы с префиксом `flickr_`

### 7. Оценка — `stage_07_evaluation`

**`ResultVisualizer`** рисует коллаж: картинка-запрос и по пять кандидатов модели / CLIP / BERT. Файл: `artifacts/<source>/compare_<имя>.png`.

```bash
python stage_07_evaluation/run.py --source flickr --image 10002456.jpg
```

Без `--image` берётся случайный запрос из файла кандидатов.

---

## Признаки пары картинок

Модель смотрит не на одну картинку, а на **пару** (запрос, кандидат):

| Признак | Смысл |
|---|---|
| `img_clip_1`, `img_clip_2` | CLIP запроса и кандидата |
| `img_bert_1`, `img_bert_2` | BERT запроса и кандидата |
| `clip_cossim` | косинус CLIP |
| `clip_l2` | евклидово расстояние CLIP |
| `bert_cossim` | косинус BERT |
| `bert_l2` | евклидово расстояние BERT |
| `target_y` | 1 — пара связанная, 0 — нет |
| `query` | id запроса (для group split в CatBoost) |

Целевая метка на Яндексе берётся из разметки related images; на Flickr пары строятся из кластеров (обучение — на Яндекс-таблице `learning_pipeline.csv`).

---

## Переменные окружения

| Переменная | Зачем |
|---|---|
| `THESIS_DATA_DIR` | корень с датасетами и старыми папками `Dataset_*` |
| `YANDEX_IMAGES_DIR` | папка JPEG Яндекса |
| `FLICKR_IMAGES_DIR` | папка Flickr30k |
| `FLICKR_CAPTIONS_CSV` | подписи Flickr (`image_name \| comment`) |
| `PROXY_URL` | прокси для скрейпера Яндекса |
| `CLIP_MODEL_ID` | HuggingFace id CLIP |
| `BERT_MODEL_ID` | HuggingFace id BERT |
| `EMBED_BATCH_SIZE` | размер батча эмбеддингов (по умолчанию 32) |
| `CLUSTERS_NUMBER` | число кластеров (1000) |
| `NEAREST_CLUSTERS_NUMBER` | сколько соседних кластеров брать (3) |
| `CANDIDATES_PER_IMAGE` | топ-N кандидатов (5) |

Пути ищутся в таком порядке: `artifacts/<source>/` → `Dataset_preparation/` → `Dataset_model/` внутри `THESIS_DATA_DIR`.

---

## Что не попадает в GitHub

Слишком большие или секретные вещи отфильтрованы `.gitignore`:

- картинки и сырые JSON/CSV (`clip_bert.json` ~ 3.4 ГБ, `learning_pipeline.csv` ~ 2.8 ГБ);
- веса моделей `*.bin`;
- `artifacts/`;
- `.env` (прокси, локальные пути).

В репозитории код, `requirements.txt`, `.env.example` и `data/README.md`.

---

## Зависимости

`torch`, `transformers`, `Pillow` — CLIP и BERT.  
`pandas`, `numpy`, `scikit-learn` — таблицы, метрики, KMeans, дерево и лес.  
`catboost`, `xgboost`, `lightgbm` — бустинг.  
`requests`, `beautifulsoup4` — сбор текстов Яндекса.  
`python-dotenv`, `tqdm`, `matplotlib`, `joblib`.

CLIP и BERT считаются на GPU, если доступен CUDA, иначе на CPU.

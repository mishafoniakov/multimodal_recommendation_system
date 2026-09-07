# Рекомендация похожих изображений: CLIP, BERT и классификатор пар

Проект на **Python**: семь этапов пайплайна. По картинке и тексту система ищет похожие изображения — в общем пространстве CLIP∥BERT, затем классификатором пар.

Данные и веса в git не входят: картинки, `clip_bert.json` и `learning_pipeline.csv` лежат локально. В репозитории — код, `requirements.txt`, `.env.example`.

---

## Зачем это репозиторий

Проект отвечает на шесть практических вопросов:

1. Как собрать карточку «картинка + текст» из двух источников (Яндекс и Flickr)?
2. Как получить вектор картинки (`CLIP`, 512) и вектор текста (`BERT` `[CLS]`, 768)?
3. Как склеить два представления по имени файла без двойного цикла?
4. Как не перебирать все пары — кластеры в пространстве CLIP∥BERT?
5. Как научить модель отличать связанную пару от случайной?
6. Что лучше на топ-5: модель, косинус CLIP или косинус BERT?

Тема диплома: **рекомендация похожих изображений**. В git нет сырых картинок и гигабайтных таблиц, только воспроизводимый код.

---

## Как устроен проект

```text
.
├── README.md
├── run.py                            общая точка входа
├── requirements.txt
├── .env.example                      пути и гиперпараметры
├── config/settings.py                Settings: пути, модели, k, top-n
├── common/                           JSON Lines, cosine / L2, логи
├── stage_01_collection/              картинки и тексты
├── stage_02_clip/                    CLIP, вектор 512
├── stage_03_bert/                    BERT `[CLS]` 768 и join с CLIP
├── stage_04_clustering/              MiniBatchKMeans и пары
├── stage_05_training/                CatBoost / XGB / LGBM / DT / RF
├── stage_06_candidates/              топ-5: модель, CLIP, BERT
├── stage_07_evaluation/              коллаж для оценки
└── data/                             локальные датасеты (не в git)
```

Читать по номерам: `stage_01` → `stage_07`. У каждого этапа свой `run.py` с блоком **«О документе»**: источник, цель, короткий результат. Прогоны пишутся в `artifacts/<source>/`.

---

## Что внутри этапов

| Папка | Вопрос | Вход | Методы |
|---|---|---|---|
| [`stage_01_collection`](stage_01_collection) | Как собрать картинку и текст? | TSV related images / Flickr CSV | `ImageCollector`, `YandexImageScraper`, `TextAggregator` |
| [`stage_02_clip`](stage_02_clip) | Какой вектор у картинки? | папка JPEG | `ClipEmbedder`, `openai/clip-vit-base-patch32` |
| [`stage_03_bert`](stage_03_bert) | Какой вектор у текста и как склеить с CLIP? | JSON Lines текстов + CLIP | `BertEmbedder`, inner join по имени файла |
| [`stage_04_clustering`](stage_04_clustering) | Как сузить кандидатов без полного перебора? | `clip_bert.json` | CLIP∥BERT (1280), `MiniBatchKMeans`, cosine / L2 |
| [`stage_05_training`](stage_05_training) | Как отличить связанную пару от случайной? | `learning_pipeline.csv` | split 60/20/20, scale на train, CatBoost и др. |
| [`stage_06_candidates`](stage_06_candidates) | Какие пять соседей у запроса? | пары + модель | топ-5 по `pred_1`, CLIP cosine, BERT cosine |
| [`stage_07_evaluation`](stage_07_evaluation) | Как сравнить три способа глазами? | кандидаты + JPEG | коллаж запроса и 5+5+5 |

Два источника:

| Источник | `--source` | Картинки | Текст |
|---|---|---|---|
| Яндекс | `yandex` | URL из TSV related images | подписи со страниц / reverse image search |
| Flickr30k | `flickr` | папка `flickr30k_images` | пять английских подписей на кадр |

### Показатели

| Код | Смысл |
|---|---|
| `img` | имя файла картинки |
| `txt` | текст карточки |
| `clip` | вектор картинки, длина 512 |
| `bert` | вектор `[CLS]`, длина 768 |
| `clip_cossim`, `clip_l2` | близость пары в CLIP |
| `bert_cossim`, `bert_l2` | близость пары в BERT |
| `target_y` | 1 — пара связанная, 0 — нет |
| `query` | id запроса (`group_id` в CatBoost) |
| `pred_1` | вероятность, что пара связанная |

---

## Единый стиль

- Python 3.10+, одна папка на этап, английские имена классов и переменных.
- Русские заголовки в README и в блоке «О документе».
- Гигабайты данных и `.env` в git не попадают.

---

## Установка

```bash
git clone https://github.com/mishafoniakov/multimodal_recommendation_system.git
cd multimodal_recommendation_system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` укажите локальные пути. Готовые файлы диплома (`clip_bert.json`, `learning_pipeline.csv`) по ~3 ГБ каждый **не копируются в git**:

```bash
THESIS_DATA_DIR=/path/to/local/datasets
YANDEX_IMAGES_DIR=/path/to/yandex/images
FLICKR_IMAGES_DIR=/path/to/flickr30k_images
FLICKR_CAPTIONS_CSV=/path/to/flickr_captions.csv
```

Прокси для сбора текстов Яндекса — только `PROXY_URL`, не в исходниках. Нужны Python 3.10+ и по возможности GPU для CLIP/BERT.

---

## Запуск

```bash
python run.py --stage <этап> --source yandex|flickr [--model catboost]
```

| `--stage` | Что делает |
|---|---|
| `collection` | собирает тексты (Flickr CSV или `result_*.json` Яндекса) |
| `clip` | считает CLIP по папке картинок |
| `bert` | считает BERT и склеивает с CLIP по имени файла |
| `cluster` | MiniBatchKMeans, пары из ближайших кластеров, cosine/L2 |
| `train` | обучение (`--model catboost\|xgboost\|lgbm\|dt\|rf`) |
| `candidates` | топ-5 кандидатов модели, CLIP и BERT |
| `eval` | коллаж запроса и 5+5+5 кандидатов |
| `all` | этапы подряд (скачивание JPEG по TSV сюда не входит) |

```bash
python run.py --stage clip --source yandex
python run.py --stage train --source yandex --model catboost
python run.py --stage candidates --source flickr
python run.py --stage eval --source yandex
```

Тот же этап из своей папки:

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

### 1. Сбор — `stage_01_collection`

**`ImageCollector`** читает TSV связанных картинок Яндекса, качает JPEG, помнит URL→файл в `dataset_dict.txt`, пишет `dataset.json` (одна строка — одна пара).

**`YandexImageScraper`** ходит в поиск Яндекса по картинке и забирает подпись плюс соседние URL. Пауза между запросами по умолчанию 15 секунд.

**`TextAggregator`**

- Яндекс: склеивает куски из `result_*.json` в `{img: txt}`;
- Flickr: читает CSV с разделителем `|`, чистит подписи и объединяет пять комментариев к одному кадру.

Выход: `image_big_text_dataset.json` (JSON Lines: `img`, `txt`).

### 2. CLIP — `stage_02_clip`

**`ClipEmbedder`** открывает картинки батчами и пишет `clip.json` / `clip_flickr.json`:

```json
{"img": "img_12_0.jpg", "clip": [0.01, -0.22, ...]}
```

Размер вектора — 512.

### 3. BERT и склейка — `stage_03_bert`

**`BertEmbedder`** токенизирует текст (`max_length=50`) и берёт вектор `[CLS]` (768 чисел).

**`EmbeddingMerger`** делает inner join CLIP и BERT **по имени файла**. Выход — `clip_bert.json`:

```json
{"img": "img_12_0.jpg", "clip": [...512...], "bert": [...768...]}
```

### 4. Кластеры — `stage_04_clustering`

**`ClusterPipeline`**

1. Склеивает CLIP∥BERT в один вектор длины 1280.
2. `MiniBatchKMeans` (по умолчанию 1000 кластеров).
3. Для каждой картинки берёт `NEAREST_CLUSTERS_NUMBER` ближайших центров (по умолчанию 3).
4. Картинки этих кластеров становятся кандидатами, кроме самой себя.
5. Для каждой пары считает `clip_cossim`, `clip_l2`, `bert_cossim`, `bert_l2` (numpy).

Выход: `nearest_clusters.json`, таблица пар, `pairs_with_metrics.csv`.

Полный перебор пар неподъёмен: кандидаты ищутся только среди соседних кластеров.

### 5. Обучение — `stage_05_training`

**`LearningDataset`** читает `learning_pipeline.csv`, делит 60 / 20 / 20, масштабирует четыре метрики близости **только по train**. Для sklearn / XGBoost / LightGBM раскладывает CLIP/BERT-списки в колонки.

| `--model` | Класс | Заметка |
|---|---|---|
| `catboost` | `CatBoostTrainer` | основная модель, `Logloss`, `scale_pos_weight=2.64`, early stopping |
| `xgboost` | `XGBoostTrainer` | 2000 деревьев, lr 0.01, depth 6 |
| `lgbm` | `LightGBMTrainer` | те же порядки гиперпараметров |
| `dt` | `DecisionTreeTrainer` | sklearn |
| `rf` | `RandomForestTrainer` | sklearn |

CatBoost дополнительно получает `group_id=query`, если колонка `query` есть в таблице.

Метрики на тесте: accuracy и ROC-AUC, график `artifacts/<source>/plots/roc_<model>.png`. Модель: `artifacts/<source>/<model>_model.bin`.

Целевая метка на Яндексе берётся из разметки related images; на Flickr пары строятся из кластеров. Обучение — на Яндекс-таблице `learning_pipeline.csv`.

### 6. Кандидаты — `stage_06_candidates`

**`CandidateRanker`** масштабирует 4 метрики, считает `P(пара связана)` по `catboost_model.bin` и для каждого запроса берёт 5 лучших по `pred_1`, по CLIP cosine и по BERT cosine.

### 7. Оценка — `stage_07_evaluation`

**`ResultVisualizer`** рисует коллаж: картинка-запрос и по пять кандидатов модели / CLIP / BERT. Без `--image` берётся случайный запрос.

---

## Данные

Локально в `THESIS_DATA_DIR` или в `data/` (папка в git пустая, кроме этого описания):

| Файл | Содержание |
|---|---|
| `image_big_text_dataset.json` | карточки `img` + `txt` |
| `clip.json` / `clip_flickr.json` | CLIP, 512 |
| `bert.json` / `bert_flickr.json` | BERT `[CLS]`, 768 |
| `clip_bert.json` / `clip_bert_flickr.json` | оба вектора на картинку |
| `learning_pipeline.csv` | таблица пар для обучения |
| `catboost_model.bin` | обученный CatBoost |

Пути ищутся в порядке: `artifacts/<source>/` → `Dataset_preparation/` → `Dataset_model/` внутри `THESIS_DATA_DIR`.

В git не попадают картинки, сырые JSON/CSV, веса `*.bin`, `artifacts/`, `.env`.

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
| `EMBED_BATCH_SIZE` | размер батча эмбеддингов (32) |
| `CLUSTERS_NUMBER` | число кластеров (1000) |
| `NEAREST_CLUSTERS_NUMBER` | сколько соседних кластеров брать (3) |
| `CANDIDATES_PER_IMAGE` | топ-N кандидатов (5) |

---

## Зависимости

`torch`, `transformers`, `Pillow` — CLIP и BERT.  
`pandas`, `numpy`, `scikit-learn` — таблицы, метрики, KMeans, дерево и лес.  
`catboost`, `xgboost`, `lightgbm` — бустинг.  
`requests`, `beautifulsoup4` — сбор текстов Яндекса.  
`python-dotenv`, `tqdm`, `matplotlib`, `joblib`.

CLIP и BERT считаются на GPU, если доступен CUDA, иначе на CPU.

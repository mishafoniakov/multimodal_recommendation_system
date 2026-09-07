# Локальные данные

Эта папка в git не входит. Сюда (или в `THESIS_DATA_DIR`) кладутся картинки и готовые векторы.

Готовые файлы диплома лежат в `Thesis_code`:

```
THESIS_DATA_DIR=.../Thesis_code
```

Ожидаемые имена:

| Файл | Содержание |
|---|---|
| `clip.json` / `clip_flickr.json` | CLIP, 512 |
| `bert.json` / `bert_flickr.json` | BERT `[CLS]`, 768 |
| `clip_bert.json` / `clip_bert_flickr.json` | оба вектора на картинку |
| `learning_pipeline.csv` | таблица пар для обучения |
| `catboost_model.bin` | обученный CatBoost |

Новые прогоны пишутся в `artifacts/yandex/` и `artifacts/flickr/`.

# Локальные данные

Эта папка в git не входит, кроме этого файла. Сюда (или в `THESIS_DATA_DIR`) кладутся картинки и готовые векторы.

```
THESIS_DATA_DIR=.../Thesis_code
```

| Файл | Содержание |
|---|---|
| `image_big_text_dataset.json` | карточки `img` + `txt` |
| `clip.json` / `clip_flickr.json` | CLIP, 512 |
| `bert.json` / `bert_flickr.json` | BERT `[CLS]`, 768 |
| `clip_bert.json` / `clip_bert_flickr.json` | оба вектора на картинку |
| `learning_pipeline.csv` | таблица пар для обучения |
| `catboost_model.bin` | обученный CatBoost |

Новые прогоны пишутся в `artifacts/yandex/` и `artifacts/flickr/`.

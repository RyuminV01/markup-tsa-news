# Markup-tsa-news

🧠 This repository contains code and tools for **Named Entity-Oriented Sentiment Analysis (TSA)** in Russian news texts using a **text-to-text generation approach** powered by [`flan-t5-tsa-thor-xl`](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-xl). The system predicts sentiment labels (positive, negative, neutral) towards named entities (persons, professions, nationalities, etc).

---

## 🔧 Technologies Used

- 🤗 [Hugging Face Transformers](https://huggingface.co/)  
- 🔍 [`flan-t5-tsa-thor-xl`](https://huggingface.co/nicolay-r/flan-t5-tsa-thor-xl) — a fine-tuned FLAN-T5 XL model for TSA  
- 🧪 [`runne_contrastive_ner`](https://github.com/bond005/runne_contrastive_ner) — used for pre-annotation and NER (запускался через Docker-контейнер)
- 🐍 Python 3.x, Jupyter Notebooks

---

## 📥 Input Format

Each sample includes:
- `sentence` — текст с контекстом
- `entity` — сущность (например, "спортсмена")
- `entity_tag` — тип сущности (PERSON, ORGANIZATION, PROFESSION, COUNTRY, NATIONALITY)
- `entity_pos_start_rel` — начальная позиция сущности в строке
- `entity_pos_end_rel` — конечная позиция сущности
- `label` — предсказанный сентимент (1 — положительно, 0 — нейтрально, -1 — отрицательно)

**Пример строки:**

| sentence                                                                                                                                  | entity     | entity\_tag | entity\_pos\_start\_rel | entity\_pos\_end\_rel | label |
| ------------------------------------------------------------------------------------------------------------------------------------------| ---------- | ----------- | ----------------------- | --------------------- | ----- |
| Джеймс «Бадди» Макгирт (James (Buddy) McGirt, тренер Дадашева упрашивал дагестанского спортсмена остановить бой, но тот хотел продолжать. | спортсмена | PROFESSION  | 86                      | 96                    | -1    |


---

## 📦 Output

Модель генерирует сентимент относительно конкретной сущности в контексте.  
Выходные данные сохраняются в формате `.tsv` с колонками, как описано выше.

---

## 🚀 Запуск

```bash
# Клонируйте репозиторий
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Установите зависимости
pip install -r requirements.txt

# Запустите ноутбук
jupyter notebook Dostal_tsdateT5.ipynb
```



## 📄 Лицензия

MIT License. Используйте свободно с указанием источника.

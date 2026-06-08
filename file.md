# Контекст для DeepSeek: помощь Марку на экзамене по ML/нейросетям

> Этот файл нужен как **контекстная инструкция для другой нейросети**.  
> Его цель — быстро объяснить, что уже было разобрано в переписке, какие есть требования к работе, какой шаблон решения нужен и как помогать Марку, когда он пришлёт новый CSV/Excel-файл с данными.

---

## 1. Кто пользователь и что ему нужно

Пользователь — **Марк**. Ему нужно подготовиться к экзаменационному заданию по анализу данных и машинному обучению.

Основная задача на экзамене: по выданному датасету выполнить полноценное ML-решение в Jupyter/аналогичной среде:

1. загрузить данные;
2. провести первичный анализ;
3. найти и обработать «грязь» в данных;
4. выполнить EDA;
5. подготовить признаки;
6. построить **2 модели**, одна из которых обязательно должна быть **нейронной сетью**;
7. оценить качество моделей;
8. сравнить результаты;
9. сохранить модель;
10. написать понятные выводы.

Код должен быть **не огромным**, а экзаменационным: компактным, понятным, но закрывающим критерии.

---

## 2. Что уже было известно из файлов и переписки

В переписке уже были загружены и изучены следующие файлы:

### 2.1. `Kod_1_1_Kriterii_otsenivania_2026 (1).xlsx`

Это файл с критериями оценивания.

По этим критериям важно закрыть следующие блоки:

- описание датасета;
- описание целевой переменной;
- первичный анализ данных;
- проверка типов признаков;
- поиск пропусков;
- обработка пропусков;
- поиск дубликатов;
- обработка дубликатов;
- анализ выбросов;
- обработка выбросов;
- исследовательский анализ данных;
- графики распределений;
- анализ target;
- boxplot-графики;
- корреляционная матрица;
- подготовка признаков;
- кодирование категориальных признаков;
- масштабирование числовых признаков;
- разделение на train/validation/test;
- построение минимум двух моделей;
- обязательная нейронная сеть;
- подбор гиперпараметров хотя бы для одной модели;
- оценка качества моделей;
- `accuracy`, `classification_report`, `confusion_matrix`;
- желательно `f1_macro`, особенно если классов несколько;
- графики обучения нейронной сети;
- сравнение моделей;
- сохранение модели;
- итоговые выводы.

---

### 2.2. `example_wine.ipynb`

Это пример «идеальной» или близкой к идеальной работы от преподавателя/из методички.

По нему поняли структуру решения:

1. загрузка библиотек;
2. загрузка датасета;
3. просмотр `.head()`, `.info()`, `.describe()`, `.shape`;
4. анализ пропусков и дубликатов;
5. работа с целевой переменной;
6. визуализация распределений;
7. обработка данных;
8. обучение моделей;
9. оценка метрик;
10. выводы.

Экзаменационное решение должно быть **похоже по логике**, но адаптировано под конкретный датасет.

---

### 2.3. `wine_exam_example_dirty.csv`

Это тестовый «грязный» датасет, который использовался для тренировки.

В нём была задача классификации вина по столбцу:

```python
WineClass
```

Также в датасете были признаки, которые нужно было удалить:

```python
SampleID
TargetCodeLeak
```

Почему:

- `SampleID` — просто идентификатор строки, не полезный признак;
- `TargetCodeLeak` — явная утечка целевой переменной, потому что фактически содержит информацию о target.

В датасете были:

- пропуски;
- дубликаты;
- грязные категориальные значения;
- числовые признаки с выбросами;
- категориальный признак вроде `AlcoholLevel`;
- целевая переменная `WineClass`.

---

### 2.4. Был создан подробный ноутбук

Уже был подготовлен подробный Jupyter Notebook:

```text
wine_exam_detailed_solution.ipynb
```

В нём была реализована полная структура решения для тестового винного датасета.

---

### 2.5. Был создан GitHub-гайд

Также был подготовлен файл:

```text
wine_exam_github_guide.md
```

Это большой подробный гайд по блокам: что делает каждый этап, зачем он нужен, какие ошибки бывают и как адаптировать решение.

---

## 3. Новые уточнения по экзамену

По словам Марка, появились новые уточнения:

1. На экзамене будет что-то вроде их личного «Google Colab» с Gemini, но Gemini могут отключить.
2. Можно будет распечатать некоторую информацию по библиотекам. Преподаватель сказал, что на обратной стороне можно напечатать код мелким шрифтом.
3. Датасет будет почти такой же, как в примере, но не на 3 категории, а на **5 категорий**.
4. В плане зашумления и грязи датасет будет похож на пример.
5. Нужно будет построить **2 модели**:
   - одну нейронную сеть обязательно;
   - вторую либо ещё одну нейронку, либо классическую модель.

Вывод: экзамен сильно облегчается, потому что задача становится шаблонной. Нужно не изобретать решение с нуля, а грамотно адаптировать готовый каркас под новый датасет.

---

## 4. Какую помощь нужно оказывать Марку, когда он отправит новый датасет

Когда Марк отправит новый файл с данными, нужно сделать следующее.

---

## 5. Сначала быстро определить структуру датасета

Нужно начать с кода:

```python
import pandas as pd
import numpy as np

path = 'file.csv'
df = pd.read_csv(path)

print(df.shape)
display(df.head())
display(df.info())
display(df.describe(include='all'))
```

Если файл Excel:

```python
df = pd.read_excel('file.xlsx')
```

Нужно понять:

- сколько строк;
- сколько столбцов;
- какие есть числовые признаки;
- какие есть категориальные признаки;
- какой столбец является target;
- есть ли ID-столбцы;
- есть ли явные утечки target;
- есть ли пропуски;
- есть ли дубликаты;
- сколько классов в target.

---

## 6. Как определить target

Target — это целевая переменная, которую нужно предсказывать.

На экзамене target может называться по-разному:

```text
WineClass
Class
target
Target
label
Label
quality
category
Type
Personality
```

Нужно проверить все столбцы:

```python
for col in df.columns:
    print(col, df[col].nunique(), df[col].dropna().unique()[:10])
```

Если один из столбцов содержит 5 категорий, скорее всего это target.

Пример:

```python
target = 'WineClass'
```

Или для нового датасета:

```python
target = 'Class'
```

Если target неочевиден, нужно сказать Марку:

> Вероятнее всего target — это столбец `...`, потому что он содержит 5 категорий и выглядит как целевая переменная.

---

## 7. Что удалить до обучения

Нужно удалить признаки, которые могут испортить обучение или дать утечку.

Обычно удаляются:

```python
cols_to_drop = [
    'SampleID',
    'ID',
    'id',
    'Index',
    'index',
    'TargetCodeLeak',
    'target_code',
    'class_code',
    'label_code'
]
```

Код:

```python
X = df.drop(columns=[target] + cols_to_drop, errors='ignore')
y = df[target]
```

Важно: если есть столбец, который почти напрямую повторяет target, его нужно удалить как утечку.

Признаки утечки:

- название содержит `target`, `label`, `class`, `code`, `encoded`;
- значение почти однозначно соответствует target;
- модель даёт подозрительно идеальное качество почти 100%.

---

## 8. Обязательная очистка данных

Перед обучением нужно сделать минимальную очистку.

### 8.1. Удалить строки без target

```python
df = df.dropna(subset=[target])
```

Объяснение:

> Строки без значения целевой переменной нельзя использовать для обучения с учителем, так как для них неизвестен правильный ответ.

---

### 8.2. Удалить дубликаты

```python
df = df.drop_duplicates()
```

Объяснение:

> Дубликаты могут искусственно усиливать влияние отдельных объектов и искажать оценку качества модели.

---

### 8.3. Очистить категориальные строки

```python
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].astype(str).str.strip()
```

Для признаков можно дополнительно привести к нижнему регистру:

```python
for col in df.drop(columns=[target], errors='ignore').select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].astype(str).str.strip().str.lower()
```

Для target лучше не всегда делать `.lower()`, чтобы не портить красивые названия классов. Но если в target есть грязь типа `class1`, `Class1`, ` class1 `, тогда можно привести к единому виду.

---

## 9. Анализ пропусков

Обязательно показать:

```python
df.isna().sum()
```

Можно в процентах:

```python
missing = df.isna().mean().sort_values(ascending=False) * 100
missing[missing > 0]
```

Объяснение:

> Был проведён анализ пропущенных значений. Пропуски в целевой переменной удаляются, а пропуски в признаках обрабатываются внутри Pipeline: числовые признаки заполняются медианой, категориальные — самым частым значением.

---

## 10. Анализ классов target

Обязательно:

```python
print(df[target].value_counts())
print(df[target].value_counts(normalize=True))
```

График:

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(data=df, x=target)
plt.title('Распределение классов целевой переменной')
plt.xticks(rotation=45)
plt.show()
```

Объяснение:

> Анализ распределения классов позволяет понять, есть ли дисбаланс. При дисбалансе одной accuracy может быть недостаточно, поэтому дополнительно используется F1-score с усреднением macro.

---

## 11. Разделение признаков на числовые и категориальные

```python
X = df.drop(columns=[target] + cols_to_drop, errors='ignore')
y = df[target]

numerical = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
categorical = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print('Числовые признаки:', numerical)
print('Категориальные признаки:', categorical)
```

---

## 12. EDA-графики, которые желательно сделать

### 12.1. Корреляционная матрица

```python
plt.figure(figsize=(10,8))
sns.heatmap(X[numerical].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица числовых признаков')
plt.show()
```

Объяснение:

> Корреляционная матрица показывает линейные взаимосвязи между числовыми признаками. Сильная корреляция может означать, что признаки несут похожую информацию.

---

### 12.2. Гистограммы числовых признаков

```python
for col in numerical:
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Распределение признака {col}')
    plt.show()
```

Объяснение:

> Гистограммы позволяют оценить форму распределения признаков, асимметрию, наличие нескольких пиков и возможных аномалий.

---

### 12.3. Boxplot для выбросов

```python
for col in numerical:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot признака {col}')
    plt.show()
```

Объяснение:

> Boxplot показывает медиану, межквартильный размах и потенциальные выбросы.

---

### 12.4. Boxplot признаков по классам

```python
for col in numerical:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x=target, y=col)
    plt.title(f'{col} по классам {target}')
    plt.xticks(rotation=45)
    plt.show()
```

Объяснение:

> Boxplot по классам помогает понять, какие признаки различаются между категориями целевой переменной и потенциально полезны для классификации.

---

## 13. Обработка выбросов

Если по критериям требуется обработка выбросов, лучше использовать IQR clipping.

```python
for col in numerical:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)
```

Объяснение:

> Выбросы были обработаны методом межквартильного размаха. Значения, выходящие за границы, не удалялись, а ограничивались допустимыми нижней и верхней границами. Это позволяет уменьшить влияние экстремальных значений и сохранить размер выборки.

Важно: если датасет маленький, лучше не удалять строки с выбросами, а именно ограничивать значения.

---

## 14. Кодирование target

Для классификации target нужно закодировать числами.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(le.classes_)
print(y_encoded[:10])
```

Если классов 5, то:

```python
num_classes = len(le.classes_)
print(num_classes)
```

Для нейронки с `categorical_crossentropy`:

```python
from tensorflow.keras.utils import to_categorical

y_train_ohe = to_categorical(y_train, num_classes=num_classes)
y_valid_ohe = to_categorical(y_valid, num_classes=num_classes)
```

---

## 15. Train/validation split

Обязательно использовать `stratify`:

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)
```

Объяснение:

> Параметр `stratify` сохраняет исходное распределение классов в обучающей и валидационной выборках. Это особенно важно при многоклассовой классификации и дисбалансе классов.

---

## 16. Pipeline предобработки

Лучший универсальный вариант:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_tr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_tr = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

prep = ColumnTransformer([
    ('num', num_tr, numerical),
    ('cat', cat_tr, categorical)
])

X_train_p = prep.fit_transform(X_train)
X_valid_p = prep.transform(X_valid)
```

Объяснение:

> Предобработка обучается только на обучающей выборке. Это предотвращает утечку данных из валидационной выборки. Числовые признаки заполняются медианой и масштабируются, категориальные признаки заполняются самым частым значением и кодируются One-Hot Encoding.

---

## 17. Важный момент для TensorFlow/Keras

После `OneHotEncoder` данные могут быть sparse matrix. Keras может упасть. Поэтому перед нейронкой лучше добавить:

```python
if hasattr(X_train_p, 'toarray'):
    X_train_nn = X_train_p.toarray()
    X_valid_nn = X_valid_p.toarray()
else:
    X_train_nn = X_train_p
    X_valid_nn = X_valid_p
```

Для RandomForest можно использовать `X_train_p`, а для нейронки — `X_train_nn`.

---

## 18. Модель 1: RandomForestClassifier

Это хороший классический вариант для экзамена.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train_p, y_train)
y_pred_rf = rf.predict(X_valid_p)

print('Accuracy:', accuracy_score(y_valid, y_pred_rf))
print('F1 macro:', f1_score(y_valid, y_pred_rf, average='macro'))
print(classification_report(y_valid, y_pred_rf, target_names=le.classes_))
print(confusion_matrix(y_valid, y_pred_rf))
```

Объяснение:

> Random Forest выбран как классическая модель машинного обучения. Он хорошо работает с табличными данными, устойчив к выбросам и способен учитывать нелинейные зависимости между признаками.

---

## 19. Подбор гиперпараметров для RandomForest

Если есть время:

```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=params,
    scoring='f1_macro',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train_p, y_train)

print(grid.best_params_)
print(grid.best_score_)

best_rf = grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_valid_p)

print('Accuracy:', accuracy_score(y_valid, y_pred_best_rf))
print('F1 macro:', f1_score(y_valid, y_pred_best_rf, average='macro'))
print(classification_report(y_valid, y_pred_best_rf, target_names=le.classes_))
```

Объяснение:

> GridSearchCV перебирает несколько комбинаций гиперпараметров и оценивает их с помощью кросс-валидации. Для многоклассовой классификации используется метрика `f1_macro`, так как она учитывает качество по каждому классу независимо от его размера.

---

## 20. Модель 2: нейронная сеть Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

num_classes = len(le.classes_)

y_train_ohe = to_categorical(y_train, num_classes=num_classes)
y_valid_ohe = to_categorical(y_valid, num_classes=num_classes)

model = keras.Sequential([
    layers.Input(shape=(X_train_nn.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train_nn,
    y_train_ohe,
    batch_size=32,
    epochs=20,
    validation_data=(X_valid_nn, y_valid_ohe),
    verbose=1
)
```

Предсказание:

```python
proba = model.predict(X_valid_nn)
y_pred_nn = np.argmax(proba, axis=1)

print('Accuracy:', accuracy_score(y_valid, y_pred_nn))
print('F1 macro:', f1_score(y_valid, y_pred_nn, average='macro'))
print(classification_report(y_valid, y_pred_nn, target_names=le.classes_))
print(confusion_matrix(y_valid, y_pred_nn))
```

Объяснение:

> Нейронная сеть состоит из входного слоя, двух скрытых полносвязных слоёв, Dropout для снижения переобучения и выходного слоя Softmax. Softmax используется для многоклассовой классификации, так как преобразует выходы сети в вероятности принадлежности к каждому классу.

---

## 21. Если TensorFlow на экзамене не работает

Если `tensorflow` недоступен, можно использовать `MLPClassifier` из sklearn. Это тоже нейронная сеть.

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train_p, y_train)
y_pred_mlp = mlp.predict(X_valid_p)

print('Accuracy:', accuracy_score(y_valid, y_pred_mlp))
print('F1 macro:', f1_score(y_valid, y_pred_mlp, average='macro'))
print(classification_report(y_valid, y_pred_mlp, target_names=le.classes_))
```

Важно: `MLPClassifier` проще и безопаснее, если среда нестабильная.

---

## 22. Confusion matrix графиком

Для RandomForest:

```python
cm = confusion_matrix(y_valid, y_pred_best_rf)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap='Blues'
)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок Random Forest')
plt.show()
```

Для нейронки:

```python
cm = confusion_matrix(y_valid, y_pred_nn)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap='Blues'
)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок нейронной сети')
plt.show()
```

---

## 23. Графики обучения нейронной сети

```python
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.title('Изменение accuracy при обучении')
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Изменение функции потерь при обучении')
plt.legend()
plt.show()
```

Объяснение:

> Если accuracy на обучении растёт, а loss падает, модель обучается. Если качество на обучающей выборке сильно лучше, чем на валидационной, это признак переобучения.

Важно: не писать заранее «переобучения нет», пока не посмотрели графики.

---

## 24. Сравнение моделей

```python
results = pd.DataFrame({
    'Model': ['Random Forest', 'Neural Network'],
    'Accuracy': [
        accuracy_score(y_valid, y_pred_best_rf),
        accuracy_score(y_valid, y_pred_nn)
    ],
    'F1_macro': [
        f1_score(y_valid, y_pred_best_rf, average='macro'),
        f1_score(y_valid, y_pred_nn, average='macro')
    ]
})

results
```

Выбор лучшей:

```python
best_model_name = results.sort_values('F1_macro', ascending=False).iloc[0]['Model']
print('Лучшая модель по F1_macro:', best_model_name)
```

Объяснение:

> Для итогового сравнения используются accuracy и F1-macro. Accuracy показывает общую долю правильных ответов, а F1-macro позволяет оценить качество модели по всем классам равномерно.

---

## 25. Сохранение моделей и предобработки

Очень важно сохранить не только модель, но и предобработчик.

```python
import joblib

joblib.dump(prep, 'preprocessor.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(best_rf, 'best_random_forest.pkl')
model.save('neural_network_model.keras')
```

Объяснение:

> Для дальнейшего использования модели необходимо сохранить не только саму модель, но и объект предобработки, а также кодировщик целевой переменной. Без них новый сырой датасет нельзя будет корректно привести к тому же виду, что и обучающие данные.

---

## 26. Финальный вывод для работы

Шаблон вывода:

```text
В ходе работы был выполнен анализ и подготовка датасета для задачи многоклассовой классификации. Были изучены типы признаков, распределение целевой переменной, наличие пропусков, дубликатов и выбросов. Пропуски в признаках были обработаны с помощью Pipeline, категориальные признаки закодированы методом One-Hot Encoding, числовые признаки стандартизированы.

Для решения задачи были построены две модели: классическая модель Random Forest и нейронная сеть. Качество моделей оценивалось с помощью accuracy, F1-macro, classification report и матрицы ошибок. По итогам сравнения лучшей была выбрана модель с наибольшим значением F1-macro, так как эта метрика лучше подходит для многоклассовой классификации и учитывает качество по каждому классу.
```

---

## 27. Что делать, если новый датасет будет на 5 классов

Нужно проверить:

```python
df[target].nunique()
df[target].value_counts()
```

Если 5 классов, то для Keras:

```python
num_classes = 5
layers.Dense(num_classes, activation='softmax')
```

Но лучше универсально:

```python
num_classes = len(le.classes_)
```

Тогда код сам подстроится под 3, 5 или любое другое количество классов.

---

## 28. Самые частые ошибки, которые нужно ловить

### Ошибка 1. Используют линейную регрессию для классификации

Если target — категории, это классификация. Нужны:

- `LogisticRegression`;
- `RandomForestClassifier`;
- `MLPClassifier`;
- Keras softmax-сеть.

Не надо использовать обычную `LinearRegression`.

---

### Ошибка 2. Забыли удалить ID

Если есть `SampleID`, `ID`, `index`, их лучше удалить.

---

### Ошибка 3. Забыли удалить утечку target

Если есть `TargetCodeLeak` или похожий столбец, удалить обязательно.

---

### Ошибка 4. Нейронка падает из-за sparse matrix

Решение:

```python
if hasattr(X_train_p, 'toarray'):
    X_train_nn = X_train_p.toarray()
    X_valid_nn = X_valid_p.toarray()
else:
    X_train_nn = X_train_p
    X_valid_nn = X_valid_p
```

---

### Ошибка 5. `classification_report` ругается на target names

Если классы не совпадают, можно убрать `target_names`:

```python
print(classification_report(y_valid, y_pred))
```

Или добавить:

```python
print(classification_report(y_valid, y_pred, target_names=le.classes_))
```

только если все классы есть в validation.

---

### Ошибка 6. Забыт `stratify`

Для классификации лучше всегда:

```python
stratify=y_encoded
```

---

### Ошибка 7. Слишком много кода

На экзамене не нужно писать промышленный проект. Нужно закрыть критерии. Лучше компактный понятный код, чем огромная система.

---

## 29. Минимальный боевой каркас, который почти всегда нужен

```python
import pandas as pd, numpy as np, random
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
```

---

## 30. Компактный порядок действий на экзамене

1. Загрузить данные.
2. Посмотреть `head`, `info`, `describe`, `shape`.
3. Найти target.
4. Удалить строки без target.
5. Удалить дубликаты.
6. Почистить строковые признаки.
7. Удалить ID и утечки.
8. Посмотреть распределение классов.
9. Разделить признаки на числовые и категориальные.
10. Построить основные графики.
11. Обработать выбросы через IQR clipping.
12. Закодировать target.
13. Сделать `train_test_split(..., stratify=y)`.
14. Сделать `Pipeline` для предобработки.
15. Обучить RandomForest.
16. Подобрать параметры через GridSearchCV.
17. Обучить нейронку.
18. Построить confusion matrix.
19. Построить графики accuracy/loss.
20. Сравнить модели в таблице.
21. Сохранить модели.
22. Написать вывод.

---

## 31. Как DeepSeek должен помогать Марку

Когда Марк пришлёт новый файл с данными, нужно:

1. Не писать абстрактный ответ.
2. Сразу посмотреть структуру данных.
3. Определить target.
4. Сказать, какие колонки удалить.
5. Сказать, какие признаки числовые/категориальные.
6. Подстроить код под реальные названия колонок.
7. Дать Марку готовые блоки кода в порядке выполнения.
8. К каждому блоку дать короткое объяснение для отчёта/экзамена.
9. Не раздувать код лишними моделями.
10. Основной вариант: `RandomForestClassifier` + `Keras Sequential`.
11. Запасной вариант: `RandomForestClassifier` + `MLPClassifier`.
12. Проверить, что для 5 классов используется `softmax` и `categorical_crossentropy`.
13. Проверить, что `num_classes = len(le.classes_)`, а не вручную захардкожено, если можно универсально.
14. Проверить, что есть `f1_macro`.
15. Проверить, что есть сохранение модели и предобработки.

---

## 32. Стиль ответа для Марка

Отвечать на русском языке.

Стиль:

- прямо;
- по делу;
- без воды;
- с объяснением, что именно менять;
- лучше блоками;
- если код может упасть, сразу указать почему;
- если есть риск потери баллов, прямо сказать.

Марк предпочитает практичные ответы: что делать, куда вставить, что заменить, что сказать на экзамене.

---

## 33. Что важно помнить про критерии

Полное решение — это не только обучение моделей.

Если есть только модели, но нет EDA/очистки/выводов — это неполно.

Минимальный набор для хорошей оценки:

- анализ данных;
- чистка данных;
- EDA;
- подготовка признаков;
- 2 модели;
- нейронка;
- метрики;
- сравнение;
- выводы.

---

## 34. Что было не так в коротком коде Марка

Марк прислал компактный код, и по нему было сказано:

Код хороший как база, но не является полным решением, потому что:

1. часть объяснений была вставлена прямо в код без `#`, из-за чего возник бы `SyntaxError`;
2. `SampleID` исключался из графиков, но не удалялся из обучения;
3. дубликаты только считались, но не удалялись;
4. выбросы только показывались на boxplot, но не обрабатывались;
5. не было явного графика распределения target;
6. не было boxplot признаков по классам;
7. не было отдельного `f1_macro`;
8. confusion matrix лучше строить графиком;
9. для Keras надо учитывать sparse matrix после OneHotEncoder;
10. сохранялась только нейронка, но не сохранялись `prep`, `le`, `best_rf`;
11. не было итоговой таблицы сравнения моделей.

Если DeepSeek будет проверять новый код Марка, нужно проверить эти же места.

---

## 35. Приоритеты, если времени на экзамене мало

Если мало времени, делать так:

### Обязательно

- загрузка данных;
- `head/info/describe`;
- target;
- пропуски;
- дубликаты;
- удаление ID/утечек;
- train/validation split;
- Pipeline;
- RandomForest;
- нейронка;
- accuracy + f1_macro + classification_report;
- confusion matrix;
- графики обучения нейронки;
- сравнение моделей;
- вывод.

### Желательно

- GridSearchCV;
- корреляционная матрица;
- boxplot по классам;
- обработка выбросов;
- сохранение модели.

### Можно пропустить, если совсем не успевает

- сложный анализ feature importance;
- ROC-AUC для многоклассовой классификации;
- сложная архитектура нейронки;
- много разных моделей;
- слишком красивые графики.

---

## 36. Универсальный шаблон объяснения для Markdown-ячеек

Можно вставлять перед кодом в Jupyter.

### Первичный анализ

```text
На данном этапе выполняется первичный анализ датасета: просматриваются первые строки, размер таблицы, типы признаков, основные статистические характеристики, наличие пропусков и дубликатов. Это необходимо для понимания структуры данных и выбора дальнейших методов обработки.
```

### Target

```text
Целевая переменная — это признак, который необходимо предсказать. В данной задаче решается задача многоклассовой классификации, так как целевая переменная содержит несколько категорий.
```

### Пропуски

```text
Пропуски в целевой переменной удаляются, так как такие объекты нельзя использовать для обучения с учителем. Пропуски в признаках обрабатываются внутри Pipeline: числовые признаки заполняются медианой, категориальные — наиболее частым значением.
```

### Предобработка

```text
Для корректной подготовки данных используется ColumnTransformer. Он позволяет применять разные преобразования к числовым и категориальным признакам. Такой подход предотвращает утечку данных, поскольку предобработка обучается только на обучающей выборке.
```

### RandomForest

```text
Random Forest используется как классическая модель машинного обучения. Модель хорошо подходит для табличных данных, устойчива к выбросам и способна учитывать нелинейные зависимости между признаками.
```

### Нейронная сеть

```text
Нейронная сеть используется как вторая модель. В выходном слое применяется Softmax, так как решается задача многоклассовой классификации. Функция потерь categorical_crossentropy подходит для обучения модели при One-Hot кодировании целевой переменной.
```

### Метрики

```text
Качество моделей оценивается с помощью accuracy, F1-macro, classification report и матрицы ошибок. Accuracy показывает общую долю правильных ответов, а F1-macro позволяет оценить качество по всем классам равномерно.
```

### Вывод

```text
После обучения и оценки моделей выполняется сравнение результатов. Лучшей считается модель, показавшая наибольшее значение F1-macro, так как данная метрика особенно важна для многоклассовой классификации и возможного дисбаланса классов.
```

---

## 37. Итоговая цель

Итоговая цель помощи Марку:

> Быстро адаптировать готовый экзаменационный шаблон под конкретный датасет, закрыть все критерии, не перегрузить код и помочь Марку понять, что именно происходит в каждом блоке.

Главное — не писать огромный ML-проект. Нужно сделать аккуратное, понятное, проверяемое решение, которое преподаватель сможет принять по критериям.

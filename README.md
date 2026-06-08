# Гайд по экзаменационному ML-решению: классификация вина на грязном датасете

Файл с решением в ноутбуке: `wine_exam_detailed_solution.ipynb`  
Тестовый датасет: `wine_exam_example_dirty.csv`

Этот markdown-файл — подробный разбор решения по блокам. Его можно использовать как README для GitHub, как конспект перед экзаменом и как основу для распечатки.

---

## 1. Что вообще делаем

В работе решается задача **многоклассовой классификации**.

Есть таблица с характеристиками вина: кислотность, содержание алкоголя, фенолы, цветовая интенсивность, пролины и другие признаки. Нужно по этим признакам определить класс вина.

В тестовом файле целевая переменная называется:

```python
WineClass
```

Классы в тестовом датасете:

```text
A
B
C
```

На экзамене, по уточнению, может быть похожий датасет, но не с 3, а с 5 категориями. Логика решения от этого почти не меняется. Меняется только количество классов в целевой переменной.

---

## 2. Главная идея решения

На экзамене важно не написать самый сложный код, а закрыть критерии:

1. Загрузить данные.
2. Изучить структуру таблицы.
3. Найти целевую переменную.
4. Проверить пропуски, дубликаты, грязные значения.
5. Удалить технические признаки и утечки.
6. Обработать выбросы.
7. Провести EDA — разведочный анализ данных.
8. Подготовить признаки и target.
9. Разделить данные на train/validation.
10. Собрать `Pipeline` для предобработки.
11. Построить минимум 2 модели.
12. Среди моделей обязательно должна быть нейронная сеть.
13. Подобрать гиперпараметры.
14. Вывести метрики.
15. Построить графики: confusion matrix, validation curve, learning curve, loss curve.
16. Сравнить модели.
17. Сохранить лучшую модель.
18. Написать итоговый вывод.

---

## 3. Почему задача именно классификация, а не регрессия

Целевая переменная `WineClass` содержит категории: `A`, `B`, `C`. Это не числа, которые нужно предсказать как величину. Это классы объектов.

Поэтому задача — **классификация**.

Важно не перепутать:

| Ситуация | Тип задачи |
|---|---|
| Предсказать цену, температуру, балл, вес | Регрессия |
| Предсказать класс, категорию, тип, метку | Классификация |

Если на экзамене говорят «линейная регрессия», для такой задачи лучше использовать не `LinearRegression`, а `LogisticRegression`, потому что логистическая регрессия — это модель классификации.

---

## 4. Что есть в тестовом датасете

В файле `wine_exam_example_dirty.csv` есть 183 строки и 17 столбцов.

Основные столбцы:

```text
SampleID
alcohol
malic_acid
ash
alcalinity_of_ash
magnesium
total_phenols
flavanoids
nonflavanoid_phenols
proanthocyanins
color_intensity
hue
od280/od315_of_diluted_wines
proline
WineClass
TargetCodeLeak
AlcoholLevel
```

Ключевые особенности грязи в датасете:

| Проблема | Где встречается | Что делаем |
|---|---|---|
| Пропуски | `alcohol`, `color_intensity`, `proline` | Заполняем внутри `Pipeline` |
| Дубликаты | Есть полные дубликаты строк | Удаляем `drop_duplicates()` |
| Грязные категории | `AlcoholLevel` содержит не только `low`, `medium`, `high`, но и мусорные значения | Заменяем мусор на `NaN` |
| Технический ID | `SampleID` | Удаляем |
| Утечка target | `TargetCodeLeak` | Обязательно удаляем |
| Выбросы | В числовых признаках | Ограничиваем методом IQR |

---

## 5. Почему `TargetCodeLeak` обязательно удалить

`TargetCodeLeak` — это признак-утечка. Он напрямую связан с целевой переменной.

Например:

```text
WineClass = A -> TargetCodeLeak = 0
WineClass = B -> TargetCodeLeak = 1
WineClass = C -> TargetCodeLeak = 2
```

Если оставить такой столбец, модель получит почти готовый ответ. Качество будет искусственно высоким, но это будет нечестное обучение.

На экзамене это важный момент: нужно показать, что ты понимаешь разницу между полезным признаком и утечкой целевой переменной.

---

## 6. Общая структура ноутбука

Ноутбук разбит на блоки:

| Блок | Назначение |
|---|---|
| 0 | Импорт библиотек |
| 1 | Загрузка данных и первичный осмотр |
| 2 | Описание target и структуры данных |
| 3 | Предобработка данных |
| 3.1 | Обработка выбросов |
| 4 | EDA |
| 5 | Формирование `X` и `y` |
| 6 | Разбиение на train/validation |
| 7 | Pipeline предобработки |
| 8 | Метрики качества |
| 9 | Модель 1: Logistic Regression |
| 10 | Подбор гиперпараметров Logistic Regression |
| 11 | Модель 2: MLPClassifier |
| 12 | Подбор гиперпараметров MLPClassifier |
| 13 | Confusion matrix |
| 14 | Validation curve |
| 15 | Learning curve |
| 16 | Loss curve нейронки |
| 17 | Сравнение моделей |
| 18 | Сохранение модели |
| 19 | Итоговый вывод |
| 20 | Мини-шпаргалка |

---

# Блок 0. Импорт библиотек

## Зачем нужен блок

В начале подключаются все библиотеки, которые понадобятся для анализа данных, построения графиков, обучения моделей и сохранения результата.

Основные библиотеки:

- `pandas` — работа с таблицами;
- `numpy` — числовые операции;
- `matplotlib` — графики;
- `sklearn` — машинное обучение;
- `joblib` — сохранение модели.

Также задаётся `RANDOM_STATE = 42`. Это нужно, чтобы разбиение данных и обучение моделей были воспроизводимыми.

## Код

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib

RANDOM_STATE = 42
```

## Что можно сказать на экзамене

> В этом блоке я импортирую библиотеки для загрузки и анализа данных, построения графиков, предобработки признаков, обучения моделей, оценки качества и сохранения итоговой модели. `RANDOM_STATE` фиксирует случайность, чтобы результаты можно было воспроизвести.

---

# Блок 1. Загрузка данных и первичный осмотр

## Зачем нужен блок

Перед обработкой данных нужно понять, что лежит в таблице:

- сколько строк и столбцов;
- какие есть признаки;
- какие типы данных;
- есть ли пропуски;
- есть ли дубликаты;
- как выглядит начало таблицы.

Это закрывает базовый критерий анализа структуры датасета.

## Код

```python
DATA_PATH = 'wine_exam_example_dirty.csv'

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    df = pd.read_csv('/mnt/data/wine_exam_example_dirty.csv')

print('Размер датасета:', df.shape)
df.head()
```

```python
print('Информация о датасете:')
df.info()
```

```python
print('Количество пропусков по столбцам:')
df.isna().sum().sort_values(ascending=False)
```

```python
print('Количество полных дубликатов:', df.duplicated().sum())
print('Количество уникальных значений по столбцам:')
df.nunique().sort_values()
```

## Что показывает этот блок

`df.shape` показывает размер таблицы.

`df.head()` показывает первые 5 строк.

`df.info()` показывает:

- названия столбцов;
- типы данных;
- количество непустых значений;
- примерный объём памяти.

`df.isna().sum()` показывает количество пропусков.

`df.duplicated().sum()` показывает количество полностью повторяющихся строк.

`df.nunique()` показывает количество уникальных значений в каждом столбце. Это помогает найти:

- ID-столбцы;
- почти константные признаки;
- категориальные признаки;
- подозрительные признаки, связанные с target.

## Что можно сказать на экзамене

> Я сначала провожу первичный осмотр данных: смотрю размер таблицы, типы признаков, пропуски, дубликаты и количество уникальных значений. Это нужно, чтобы понять качество данных и составить план предобработки.

---

# Блок 2. Описание структуры датасета и целевой переменной

## Зачем нужен блок

Нужно явно определить, какой столбец является целевой переменной.

В тестовом датасете это:

```python
TARGET = 'WineClass'
```

Target показывает класс вина. Так как классов несколько, решается задача многоклассовой классификации.

## Код

```python
TARGET = 'WineClass'

print('Целевая переменная:', TARGET)
print('Классы целевой переменной:')
print(df[TARGET].value_counts().sort_index())

print('\nДоли классов:')
print((df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2).astype(str) + '%')
```

```python
numeric_cols_initial = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols_initial = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print('Числовые столбцы:', numeric_cols_initial)
print('Категориальные столбцы:', categorical_cols_initial)
```

## Почему смотрим доли классов

Если один класс встречается намного чаще других, датасет несбалансирован. Тогда обычная accuracy может быть обманчивой.

Например, если 90% объектов относятся к классу `B`, модель может всегда предсказывать `B` и получить accuracy 90%, но реального качества не будет.

Поэтому в работе используется `f1_macro`, потому что эта метрика учитывает качество по каждому классу отдельно.

## Что можно сказать на экзамене

> Целевая переменная — `WineClass`, она содержит классы вина. Так как значений несколько, это задача многоклассовой классификации. Я проверяю распределение классов, чтобы понять, есть ли дисбаланс, и поэтому дальше использую не только accuracy, но и F1 macro.

---

# Блок 3. Предобработка данных

## Зачем нужен блок

Сырые данные почти всегда содержат проблемы. В этом датасете есть:

- дубликаты;
- грязные категориальные значения;
- пропуски;
- выбросы;
- технический идентификатор;
- признак-утечка.

Предобработка нужна, чтобы модель обучалась на корректных признаках.

---

## 3.1. Создание копии и удаление дубликатов

### Зачем делаем копию

Исходный `df` лучше не менять напрямую. Создаём `clean_df`, чтобы всегда можно было вернуться к исходным данным.

### Код

```python
clean_df = df.copy()

print('Строк до удаления дубликатов:', clean_df.shape[0])
print('Полных дубликатов:', clean_df.duplicated().sum())

clean_df = clean_df.drop_duplicates()

print('Строк после удаления дубликатов:', clean_df.shape[0])
```

### Почему удаляем дубликаты

Дубликаты могут завысить качество модели. Если одинаковая строка попадёт и в train, и в validation, модель фактически увидит ответ заранее.

На маленьких датасетах дубликаты особенно опасны.

---

## 3.2. Очистка категориальных признаков

### Зачем нужна очистка

В категориальных признаках часто встречаются:

- лишние пробелы;
- разные регистры;
- пустые строки;
- `?`, `none`, `nan`;
- мусорные значения.

В этом датасете специально зашумлён столбец `AlcoholLevel`.

Правильные значения:

```text
low
medium
high
```

Мусорные значения:

```text
42
777
999
```

Их нужно заменить на `NaN`, чтобы потом заполнить самым частым значением внутри `Pipeline`.

### Код

```python
for col in clean_df.select_dtypes(include=['object', 'category']).columns:
    if col != TARGET:
        clean_df[col] = (
            clean_df[col]
            .astype('string')
            .str.strip()
            .str.lower()
            .replace({'': np.nan, 'nan': np.nan, 'none': np.nan, '?': np.nan})
        )

if 'AlcoholLevel' in clean_df.columns:
    allowed_levels = ['low', 'medium', 'high']
    before_values = clean_df['AlcoholLevel'].value_counts(dropna=False)
    clean_df['AlcoholLevel'] = clean_df['AlcoholLevel'].where(
        clean_df['AlcoholLevel'].isin(allowed_levels),
        np.nan
    )
    after_values = clean_df['AlcoholLevel'].value_counts(dropna=False)
    
    print('AlcoholLevel до очистки:')
    print(before_values)
    print('\nAlcoholLevel после очистки:')
    print(after_values)
else:
    print('Столбец AlcoholLevel отсутствует')
```

### Почему target не трогаем

В коде есть условие:

```python
if col != TARGET:
```

Target не приводится к нижнему регистру и не очищается как обычный категориальный признак, потому что это целевая переменная. Её мы отдельно закодируем через `LabelEncoder`.

---

## 3.3. Удаление технических признаков и утечек

### Что удаляем

```text
SampleID
TargetCodeLeak
```

`SampleID` — просто номер объекта. Он не несёт содержательной информации о вине.

`TargetCodeLeak` — утечка целевой переменной. Его обязательно нужно удалить.

### Код

```python
cols_to_drop = []

for col in ['SampleID', 'TargetCodeLeak']:
    if col in clean_df.columns:
        cols_to_drop.append(col)

clean_df = clean_df.drop(columns=cols_to_drop)

print('Удалённые признаки:', cols_to_drop)
print('Размер таблицы после удаления лишних признаков:', clean_df.shape)
```

### Что можно сказать на экзамене

> Я удаляю `SampleID`, потому что это технический идентификатор, и `TargetCodeLeak`, потому что это утечка целевой переменной. Если оставить утечку, модель получит искусственно высокое качество, но такое решение будет некорректным.

---

# Блок 3.1. Анализ и обработка выбросов

## Что такое выбросы

Выбросы — это значения, которые сильно отличаются от основной массы данных.

Например, если у большинства вин `proline` находится примерно в диапазоне 300–1500, а где-то стоит 9999, это может быть ошибка измерения или специально добавленный шум.

## Почему выбросы важны

Выбросы могут сильно влиять на модели, особенно на модели, чувствительные к масштабу признаков:

- логистическая регрессия;
- нейронная сеть;
- линейные модели;
- методы на расстояниях.

## Метод IQR

IQR — межквартильный размах.

Формулы:

```text
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

Где:

- `Q1` — первый квартиль, 25% значений ниже него;
- `Q3` — третий квартиль, 75% значений ниже него;
- `IQR` — расстояние между Q1 и Q3.

Всё, что ниже `lower_bound` или выше `upper_bound`, считается выбросом.

---

## 3.1.1. Отчёт по выбросам

### Код

```python
def make_iqr_report(data, columns, factor=1.5):
    report = []
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        outliers = ((data[col] < lower) | (data[col] > upper)).sum()
        report.append({
            'feature': col,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower,
            'upper_bound': upper,
            'outliers_count': int(outliers)
        })
    return pd.DataFrame(report)

numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
outlier_report_before = make_iqr_report(clean_df, numeric_cols)

outlier_report_before.sort_values('outliers_count', ascending=False)
```

---

## 3.1.2. Ограничение выбросов

В решении выбросы не удаляются, а ограничиваются границами IQR.

Это называется:

```text
clipping
```

или

```text
winsorization
```

Плюс такого подхода: мы не теряем строки, что полезно на маленьком датасете.

### Код

```python
def cap_outliers_iqr(data, columns, factor=1.5):
    result = data.copy()
    limits = {}
    
    for col in columns:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        
        if pd.isna(iqr) or iqr == 0:
            continue
        
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        limits[col] = (lower, upper)
        result[col] = result[col].clip(lower=lower, upper=upper)
    
    return result, limits

clean_df, outlier_limits = cap_outliers_iqr(clean_df, numeric_cols)

outlier_report_after = make_iqr_report(clean_df, numeric_cols)
print('Количество признаков, для которых применялись границы IQR:', len(outlier_limits))
outlier_report_after.sort_values('outliers_count', ascending=False)
```

### Проверка пропусков после очистки

```python
print('Пропуски после очистки грязных значений:')
clean_df.isna().sum().sort_values(ascending=False)
```

## Что можно сказать на экзамене

> Для числовых признаков я проверяю выбросы методом IQR. Сильно выбивающиеся значения не удаляю, а ограничиваю границами, чтобы сохранить размер выборки. Это уменьшает влияние экстремальных значений на модели.

---

# Блок 4. EDA — разведочный анализ данных

## Зачем нужен EDA

EDA нужен, чтобы визуально изучить данные и показать закономерности.

На экзамене EDA закрывает важный критерий: нужно не просто обучить модель, а показать понимание данных.

В EDA строятся:

1. распределение целевой переменной;
2. гистограммы/плотности числовых признаков;
3. boxplot-графики;
4. корреляционная матрица;
5. средние значения признаков по классам;
6. графики влияния признаков на target.

---

## 4.1. Распределение классов

### Код

```python
plt.figure(figsize=(6, 4))
clean_df[TARGET].value_counts().sort_index().plot(kind='bar')
plt.title('Распределение объектов по классам')
plt.xlabel('Класс вина')
plt.ylabel('Количество объектов')
plt.grid(axis='y')
plt.show()
```

### Что показывает график

График показывает, сколько объектов относится к каждому классу.

Если классы распределены примерно равномерно, accuracy можно использовать как дополнительную метрику.

Если классы несбалансированы, основную роль должна играть `f1_macro`.

---

## 4.2. Плотности распределения числовых признаков

### Код

```python
numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()

clean_df[numeric_cols].hist(figsize=(14, 10), bins=20, density=True)
plt.suptitle('Плотности распределения числовых признаков')
plt.tight_layout()
plt.show()
```

### Что значит `density=True`

Обычная гистограмма показывает количество объектов в каждом интервале.

`density=True` показывает плотность распределения. Это удобно, когда нужно сравнивать форму распределения признаков.

---

## 4.3. Boxplot числовых признаков

### Код

```python
plt.figure(figsize=(14, 6))
clean_df[numeric_cols].boxplot(rot=45)
plt.title('Boxplot числовых признаков после обработки выбросов')
plt.ylabel('Значения признаков')
plt.grid(axis='y')
plt.show()
```

### Что показывает boxplot

Boxplot показывает:

- медиану;
- межквартильный размах;
- потенциальные выбросы;
- разброс значений признаков.

После обработки выбросов график должен выглядеть спокойнее, без чрезмерно далёких значений.

---

## 4.4. Корреляционная матрица

### Код

```python
corr = clean_df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect='auto')
plt.colorbar(label='Корреляция')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Корреляционная матрица числовых признаков')
plt.tight_layout()
plt.show()
```

### Что такое корреляция

Корреляция показывает линейную связь между признаками.

Значения:

| Значение | Смысл |
|---|---|
| Близко к 1 | Сильная прямая связь |
| Близко к -1 | Сильная обратная связь |
| Близко к 0 | Линейной связи почти нет |

Если два признака сильно коррелируют, они могут частично дублировать друг друга.

---

## 4.5. Средние значения признаков по классам

### Код

```python
class_means = clean_df.groupby(TARGET)[numeric_cols].mean().T
class_means
```

```python
class_means.plot(kind='bar', figsize=(14, 6))
plt.title('Средние значения числовых признаков по классам')
plt.xlabel('Признак')
plt.ylabel('Среднее значение')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

### Что показывает этот блок

Если средние значения признаков заметно отличаются между классами, значит признаки полезны для классификации.

Например, если у класса `A` среднее значение `alcohol` выше, чем у классов `B` и `C`, этот признак может помогать модели отличать класс `A`.

---

## 4.6. Boxplot нескольких важных признаков по target

### Код

```python
important_features = [col for col in ['alcohol', 'flavanoids', 'color_intensity', 'proline'] if col in clean_df.columns]

for col in important_features:
    plt.figure(figsize=(7, 4))
    clean_df.boxplot(column=col, by=TARGET)
    plt.title(f'{col} по классам вина')
    plt.suptitle('')
    plt.xlabel('Класс вина')
    plt.ylabel(col)
    plt.grid(axis='y')
    plt.show()
```

### Зачем этот график

Он показывает, как конкретный признак распределяется внутри каждого класса.

Это сильнее, чем просто среднее значение, потому что видно:

- медиану;
- разброс;
- перекрытие классов;
- возможные выбросы.

---

## Краткий вывод по EDA

Пример вывода:

> На этапе EDA было показано, что классы вина представлены в датасете не идеально равномерно, поэтому для оценки качества используется `f1_macro`. Числовые признаки имеют разные масштабы и распределения, поэтому требуется стандартизация. Некоторые признаки заметно различаются между классами, значит они могут быть полезны для классификации.

---

# Блок 5. Формирование признаков и целевой переменной

## Зачем нужен блок

Модели отдельно получают:

- `X` — признаки;
- `y` — целевую переменную.

Из `X` нужно удалить target, чтобы модель не обучалась на правильном ответе.

## Код

```python
X = clean_df.drop(columns=[TARGET])
y_raw = clean_df[TARGET]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print('Классы после кодирования:')
for code, label in enumerate(label_encoder.classes_):
    print(code, '->', label)

print('Размер X:', X.shape)
print('Размер y:', y.shape)
```

## Зачем нужен `LabelEncoder`

Многие модели sklearn ожидают, что классы будут закодированы числами.

Например:

```text
A -> 0
B -> 1
C -> 2
```

Если на экзамене будет 5 классов, будет примерно так:

```text
class_1 -> 0
class_2 -> 1
class_3 -> 2
class_4 -> 3
class_5 -> 4
```

Логика не меняется.

---

# Блок 6. Разбиение на обучающую и валидационную выборки

## Зачем делить данные

Модель нужно проверять на данных, которые она не видела при обучении.

Поэтому данные делятся на:

- `train` — для обучения;
- `validation` — для проверки качества.

## Код

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print('Train:', X_train.shape)
print('Validation:', X_valid.shape)
```

## Почему нужен `stratify=y`

`stratify=y` сохраняет пропорции классов в train и validation.

Это особенно важно при многоклассовой классификации.

Без `stratify` может случиться так, что один из редких классов почти исчезнет из validation-выборки.

## Что можно сказать на экзамене

> Я использую стратифицированное разбиение, чтобы соотношение классов в обучающей и валидационной выборках осталось примерно таким же, как в исходном датасете.

---

# Блок 7. Pipeline предобработки

## Зачем нужен Pipeline

`Pipeline` нужен, чтобы все преобразования данных выполнялись правильно и одинаково:

- на train;
- на validation;
- на новых данных после сохранения модели.

Главный плюс: `Pipeline` помогает избежать утечки данных.

## Что делаем с числовыми признаками

Для числовых признаков:

1. Заполняем пропуски медианой.
2. Стандартизируем значения.

```python
SimpleImputer(strategy='median')
StandardScaler()
```

Почему медиана: она устойчивее к выбросам, чем среднее.

Почему стандартизация: логистическая регрессия и нейронная сеть чувствительны к масштабу признаков.

## Что делаем с категориальными признаками

Для категориальных признаков:

1. Заполняем пропуски самым частым значением.
2. Кодируем через One-Hot Encoding.

```python
SimpleImputer(strategy='most_frequent')
OneHotEncoder(handle_unknown='ignore')
```

`handle_unknown='ignore'` нужен, чтобы модель не падала, если в новых данных появится категория, которой не было при обучении.

## Код

```python
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()

print('Числовые признаки:', numeric_features)
print('Категориальные признаки:', categorical_features)
```

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```

## Что можно сказать на экзамене

> Для числовых признаков я заполняю пропуски медианой и масштабирую данные. Для категориальных признаков заполняю пропуски самым частым значением и применяю one-hot encoding. Всё это помещено в `Pipeline`, чтобы преобразования обучались только на train-выборке и не было утечки данных.

---

# Блок 8. Метрики качества

## Почему одной accuracy мало

Accuracy показывает долю правильных ответов.

Но если классы несбалансированы, accuracy может обманывать.

Поэтому в работе основная метрика:

```python
f1_macro
```

Она считает F1 отдельно по каждому классу, а потом усредняет.

## Код функции оценки

```python
def evaluate_model(model_name, model, X_valid, y_valid, label_encoder):
    y_pred = model.predict(X_valid)
    
    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='macro')
    
    print('=' * 70)
    print(model_name)
    print('Accuracy:', round(acc, 4))
    print('F1 macro:', round(f1, 4))
    print('\nClassification report:')
    print(classification_report(
        y_valid,
        y_pred,
        target_names=label_encoder.classes_
    ))
    
    return {
        'model_name': model_name,
        'accuracy': acc,
        'f1_macro': f1,
        'y_pred': y_pred
    }
```

## Что такое `classification_report`

Он показывает:

| Метрика | Смысл |
|---|---|
| precision | Насколько точны предсказания класса |
| recall | Сколько объектов класса модель нашла |
| f1-score | Баланс precision и recall |
| support | Сколько объектов этого класса было |

## Что можно сказать на экзамене

> Для оценки я использую accuracy и F1 macro. Основной метрикой выбираю F1 macro, потому что задача многоклассовая, и важно учитывать качество по каждому классу, а не только общую долю правильных ответов.

---

# Блок 9. Модель 1 — Logistic Regression

## Почему выбрана логистическая регрессия

`LogisticRegression` — классическая модель классификации.

Плюсы:

- быстро обучается;
- хорошо работает как базовая модель;
- подходит для многоклассовой классификации;
- даёт понятный baseline.

Важно: это не линейная регрессия. Несмотря на слово `Regression`, модель используется для классификации.

## Код

```python
log_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

log_reg.fit(X_train, y_train)
result_lr = evaluate_model('Logistic Regression', log_reg, X_valid, y_valid, label_encoder)
```

## Зачем `class_weight='balanced'`

Этот параметр помогает, если классы представлены неравномерно.

Модель будет сильнее учитывать менее частые классы.

---

# Блок 10. Оптимизация Logistic Regression

## Зачем нужен подбор гиперпараметров

Гиперпараметры задаются до обучения модели. Они влияют на качество.

Для логистической регрессии подбираем:

- `C` — сила регуляризации;
- `solver` — алгоритм оптимизации.

## Код

```python
param_grid_lr = {
    'model__C': [0.1, 1, 10],
    'model__solver': ['lbfgs']
}

grid_lr = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid_lr,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1
)

grid_lr.fit(X_train, y_train)

print('Лучшие параметры Logistic Regression:')
print(grid_lr.best_params_)
print('Лучший F1 macro на CV:', round(grid_lr.best_score_, 4))

result_lr_opt = evaluate_model('Optimized Logistic Regression', grid_lr.best_estimator_, X_valid, y_valid, label_encoder)
```

## Почему `scoring='f1_macro'`

Мы выбираем параметры не по accuracy, а по `f1_macro`, потому что именно эта метрика лучше подходит для многоклассовой задачи.

## Почему сетка маленькая

На экзамене время ограничено. Поэтому лучше взять небольшую сетку, которая быстро работает и формально закрывает критерий подбора гиперпараметров.

---

# Блок 11. Модель 2 — нейронная сеть MLPClassifier

## Почему MLPClassifier

`MLPClassifier` — это многослойный перцептрон, то есть простая полносвязная нейронная сеть.

Плюсы для экзамена:

- это нейронная модель;
- не нужен TensorFlow/Keras;
- работает прямо в sklearn;
- легко включается в `Pipeline`;
- удобно строить `loss_curve_`.

## Код

```python
mlp = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE
    ))
])

mlp.fit(X_train, y_train)
result_mlp = evaluate_model('MLPClassifier', mlp, X_valid, y_valid, label_encoder)
```

## Пояснение параметров

| Параметр | Что значит |
|---|---|
| `hidden_layer_sizes=(64, 32)` | Два скрытых слоя: 64 и 32 нейрона |
| `activation='relu'` | Функция активации ReLU |
| `alpha=0.001` | L2-регуляризация |
| `learning_rate_init=0.001` | Начальная скорость обучения |
| `max_iter=500` | Максимальное число эпох |
| `early_stopping=True` | Ранняя остановка при ухудшении качества |

## Что можно сказать на экзамене

> В качестве нейронной модели я использую `MLPClassifier`, то есть многослойный перцептрон. Он подходит для табличных данных и позволяет закрыть требование построения нейронной сети без отдельного подключения TensorFlow.

---

# Блок 12. Оптимизация нейронной сети

## Что подбираем

Для нейронной сети подбираем:

- архитектуру скрытых слоёв;
- коэффициент регуляризации `alpha`;
- скорость обучения.

## Код

```python
param_grid_mlp = {
    'model__hidden_layer_sizes': [(32,), (64, 32)],
    'model__alpha': [0.0001, 0.001],
    'model__learning_rate_init': [0.001]
}

grid_mlp = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid_mlp,
    cv=2,
    scoring='f1_macro',
    n_jobs=1
)

grid_mlp.fit(X_train, y_train)

print('Лучшие параметры MLPClassifier:')
print(grid_mlp.best_params_)
print('Лучший F1 macro на CV:', round(grid_mlp.best_score_, 4))

result_mlp_opt = evaluate_model('Optimized MLPClassifier', grid_mlp.best_estimator_, X_valid, y_valid, label_encoder)
```

## Почему `cv=2`, а не больше

На экзамене датасет может быть небольшой, а время ограничено. Для нейронной сети `GridSearchCV` может работать долго.

`cv=2` — компромисс: подбор гиперпараметров есть, но он не занимает слишком много времени.

---

# Блок 13. Матрицы ошибок

## Что такое confusion matrix

Матрица ошибок показывает, какие классы модель предсказывает правильно, а какие путает.

По диагонали стоят правильные предсказания.

Вне диагонали — ошибки.

## Код

```python
for title, y_pred in [
    ('Optimized Logistic Regression', result_lr_opt['y_pred']),
    ('Optimized MLPClassifier', result_mlp_opt['y_pred'])
]:
    cm = confusion_matrix(y_valid, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )
    disp.plot()
    plt.title(f'Матрица ошибок: {title}')
    plt.show()
```

## Что можно сказать на экзамене

> Матрица ошибок позволяет увидеть не только общую точность, но и то, какие классы модель чаще всего путает между собой.

---

# Блок 14. Кривая валидации

## Зачем нужна validation curve

Кривая валидации показывает, как меняется качество модели при разных значениях одного гиперпараметра.

В решении рассматривается параметр `alpha` у MLPClassifier.

## Как читать график

| Ситуация | Вывод |
|---|---|
| Train высокий, validation низкий | Переобучение |
| Train и validation оба низкие | Недообучение |
| Train и validation близкие и достаточно высокие | Модель подобрана нормально |

## Код

```python
param_range_alpha = np.array([0.0001, 0.001, 0.01])

train_scores, valid_scores = validation_curve(
    estimator=Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE
        ))
    ]),
    X=X_train,
    y=y_train,
    param_name='model__alpha',
    param_range=param_range_alpha,
    cv=2,
    scoring='f1_macro',
    n_jobs=1
)

plt.figure(figsize=(8, 5))
plt.semilogx(param_range_alpha, train_scores.mean(axis=1), marker='o', label='Train')
plt.semilogx(param_range_alpha, valid_scores.mean(axis=1), marker='o', label='Validation')
plt.title('Кривая валидации MLPClassifier по alpha')
plt.xlabel('alpha')
plt.ylabel('F1 macro')
plt.legend()
plt.grid(True)
plt.show()
```

---

# Блок 15. Кривая обучения

## Зачем нужна learning curve

Кривая обучения показывает, как качество меняется при увеличении количества обучающих данных.

Она помогает понять:

- хватает ли данных;
- переобучается ли модель;
- улучшается ли качество при добавлении данных.

## Код

```python
train_sizes, train_scores_lc, valid_scores_lc = learning_curve(
    estimator=grid_mlp.best_estimator_,
    X=X_train,
    y=y_train,
    cv=2,
    scoring='f1_macro',
    n_jobs=1,
    train_sizes=np.linspace(0.3, 1.0, 4)
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores_lc.mean(axis=1), marker='o', label='Train')
plt.plot(train_sizes, valid_scores_lc.mean(axis=1), marker='o', label='Validation')
plt.title('Кривая обучения лучшей MLPClassifier')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('F1 macro')
plt.legend()
plt.grid(True)
plt.show()
```

## Что можно сказать на экзамене

> Кривая обучения показывает, как модель ведёт себя при разном размере обучающей выборки. Если между train и validation большой разрыв, есть переобучение. Если обе кривые низкие, модель недообучается.

---

# Блок 16. График функции потерь нейронной сети

## Что такое loss curve

`loss_curve_` показывает, как ошибка нейронной сети менялась по эпохам.

Если loss постепенно уменьшается, обучение идёт нормально.

Если loss не уменьшается или сильно скачет, могут быть проблемы:

- слишком большая скорость обучения;
- плохая архитектура;
- признаки не масштабированы;
- слишком мало эпох;
- данные слишком шумные.

## Код

```python
best_mlp_model = grid_mlp.best_estimator_.named_steps['model']

plt.figure(figsize=(8, 5))
plt.plot(best_mlp_model.loss_curve_)
plt.title('График функции потерь лучшей MLPClassifier')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
```

---

# Блок 17. Сравнение моделей и выбор лучшей

## Зачем сравнивать модели

Нужно показать, какая модель работает лучше.

Сравнение делается по:

- `accuracy`;
- `f1_macro`.

Основная метрика — `f1_macro`.

## Код

```python
results = pd.DataFrame([
    {k: v for k, v in result_lr.items() if k != 'y_pred'},
    {k: v for k, v in result_lr_opt.items() if k != 'y_pred'},
    {k: v for k, v in result_mlp.items() if k != 'y_pred'},
    {k: v for k, v in result_mlp_opt.items() if k != 'y_pred'},
])

results = results.sort_values('f1_macro', ascending=False).reset_index(drop=True)
results
```

```python
plt.figure(figsize=(9, 5))
plt.bar(results['model_name'], results['f1_macro'])
plt.title('Сравнение моделей по F1 macro')
plt.xlabel('Модель')
plt.ylabel('F1 macro')
plt.xticks(rotation=25, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()
```

```python
if result_mlp_opt['f1_macro'] >= result_lr_opt['f1_macro']:
    best_model = grid_mlp.best_estimator_
    best_model_name = 'optimized_mlp_classifier'
    best_score = result_mlp_opt['f1_macro']
else:
    best_model = grid_lr.best_estimator_
    best_model_name = 'optimized_logistic_regression'
    best_score = result_lr_opt['f1_macro']

print('Лучшая модель:', best_model_name)
print('F1 macro лучшей модели:', round(best_score, 4))
```

## Что можно сказать на экзамене

> Я сравниваю модели по F1 macro и выбираю лучшую модель по этой метрике, потому что она учитывает качество классификации по каждому классу.

---

# Блок 18. Сохранение итоговой модели

## Зачем сохранять модель

После обучения лучшую модель нужно сохранить, чтобы потом использовать её без повторного обучения.

Важно сохранить не только модель, но и `LabelEncoder`, потому что он нужен для обратного перевода числовых классов в исходные названия.

## Код

```python
joblib.dump(best_model, 'best_exam_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print('Модель сохранена в файл best_exam_model.pkl')
print('Кодировщик классов сохранён в файл label_encoder.pkl')
```

## Проверка загрузки

```python
loaded_model = joblib.load('best_exam_model.pkl')
loaded_encoder = joblib.load('label_encoder.pkl')

sample_pred_encoded = loaded_model.predict(X_valid.head(5))
sample_pred_labels = loaded_encoder.inverse_transform(sample_pred_encoded)

print('Предсказания для первых 5 объектов validation:')
print(sample_pred_labels)
```

## Почему модель сохраняется вместе с Pipeline

`best_model` — это не просто модель. Это весь Pipeline:

```text
предобработка -> модель
```

Поэтому при загрузке модель сама выполнит:

- заполнение пропусков;
- масштабирование числовых признаков;
- кодирование категорий;
- предсказание класса.

---

# Блок 19. Итоговый вывод для отчёта

Готовый текст, который можно вставить в ноутбук или отчёт:

```text
В работе была решена задача многоклассовой классификации вина.

Была выполнена предобработка данных: удалены дубликаты, очищен категориальный признак AlcoholLevel, исключены технический идентификатор SampleID и признак TargetCodeLeak, являющийся утечкой целевой переменной. Для числовых признаков была проведена обработка выбросов методом IQR. Пропущенные значения обрабатывались внутри Pipeline: числовые признаки заполнялись медианой, категориальные — самым частым значением.

На этапе EDA были построены графики распределения целевой переменной, плотности распределения числовых признаков, boxplot-графики, корреляционная матрица и визуализация различий признаков между классами. Было показано, что признаки имеют разные масштабы и распределения, поэтому стандартизация необходима для логистической регрессии и нейронной сети.

Для прогнозирования были построены две модели: логистическая регрессия и нейронная сеть MLPClassifier. Для обеих моделей выполнен подбор гиперпараметров через GridSearchCV. Качество оценивалось с помощью accuracy, classification_report и основной метрики f1_macro, подходящей для многоклассовой классификации.

Дополнительно были построены матрицы ошибок, кривая валидации, кривая обучения и график функции потерь нейронной сети. По результатам сравнения была выбрана лучшая модель по f1_macro, после чего она была сохранена в файл для дальнейшего использования.
```

---

# Как адаптировать это решение под экзаменационный датасет с 5 классами

## 1. Поменять путь к файлу

```python
DATA_PATH = 'your_exam_dataset.csv'
df = pd.read_csv(DATA_PATH)
```

## 2. Поменять target

Если целевой столбец называется иначе:

```python
TARGET = 'название_целевого_столбца'
```

Например:

```python
TARGET = 'Class'
```

или

```python
TARGET = 'Category'
```

## 3. Проверить классы

```python
df[TARGET].value_counts()
```

Если классов 5, это нормально. `LabelEncoder` сам закодирует их в `0, 1, 2, 3, 4`.

## 4. Проверить мусорные столбцы

На экзамене могут быть похожие проблемы:

- `ID`;
- `SampleID`;
- `Unnamed: 0`;
- `TargetCode`;
- `TargetCodeLeak`;
- столбец, напрямую повторяющий target числом.

Такие признаки нужно удалить.

Пример универсального удаления:

```python
cols_to_drop = []

for col in ['ID', 'Id', 'id', 'SampleID', 'Unnamed: 0', 'TargetCodeLeak']:
    if col in clean_df.columns:
        cols_to_drop.append(col)

clean_df = clean_df.drop(columns=cols_to_drop)
```

## 5. Проверить грязные категории

```python
for col in clean_df.select_dtypes(include=['object', 'category']).columns:
    print(col)
    print(clean_df[col].value_counts(dropna=False))
    print()
```

Если видишь странные значения, например:

```text
?
none
unknown
999
777
42
ошибка
```

их можно заменить на `NaN`.

## 6. Количество классов в sklearn менять не нужно

Если используешь `LogisticRegression` и `MLPClassifier`, вручную указывать число классов не надо.

Модели сами поймут количество классов по `y_train`.

## 7. Если используешь Keras/TensorFlow

Тогда для 5 классов последний слой должен быть:

```python
Dense(5, activation='softmax')
```

А loss:

```python
loss='sparse_categorical_crossentropy'
```

Но для экзамена проще и безопаснее использовать `MLPClassifier` из sklearn.

---

# Типовые ошибки на экзамене

## Ошибка 1. Использовать LinearRegression для классификации

Плохо:

```python
from sklearn.linear_model import LinearRegression
```

Для классов так делать не нужно.

Хорошо:

```python
from sklearn.linear_model import LogisticRegression
```

---

## Ошибка 2. Не удалить target из X

Плохо:

```python
X = clean_df
```

Хорошо:

```python
X = clean_df.drop(columns=[TARGET])
```

---

## Ошибка 3. Оставить признак-утечку

Плохо:

```python
X = clean_df.drop(columns=[TARGET])
# TargetCodeLeak остался внутри X
```

Хорошо:

```python
clean_df = clean_df.drop(columns=['TargetCodeLeak'])
X = clean_df.drop(columns=[TARGET])
```

---

## Ошибка 4. Заполнить пропуски до train_test_split по всей таблице

Не критически смертельно для простого экзамена, но методически хуже.

Лучше делать заполнение внутри Pipeline, чтобы значения для заполнения считались только по train-выборке.

---

## Ошибка 5. Забыть `stratify=y`

Плохо:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

Хорошо:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## Ошибка 6. Оценивать только accuracy

Плохо:

```python
accuracy_score(y_valid, y_pred)
```

Хорошо:

```python
accuracy_score(y_valid, y_pred)
f1_score(y_valid, y_pred, average='macro')
classification_report(y_valid, y_pred)
```

---

## Ошибка 7. Не масштабировать признаки для нейронки

Нейронная сеть плохо работает, если признаки в разных масштабах.

Например:

```text
alcohol ~ 12
proline ~ 1000
hue ~ 1
```

Поэтому нужен:

```python
StandardScaler()
```

---

# Минимальная шпаргалка по порядку действий

1. Загрузить данные.
2. Посмотреть `head`, `info`, `shape`.
3. Проверить пропуски.
4. Проверить дубликаты.
5. Определить target.
6. Посмотреть распределение target.
7. Удалить ID и утечки.
8. Очистить грязные категории.
9. Проверить выбросы.
10. Построить EDA-графики.
11. Разделить `X` и `y`.
12. Закодировать target через `LabelEncoder`.
13. Сделать `train_test_split(..., stratify=y)`.
14. Собрать `ColumnTransformer`.
15. Обучить `LogisticRegression`.
16. Обучить `MLPClassifier`.
17. Сделать `GridSearchCV`.
18. Вывести `classification_report`.
19. Построить confusion matrix.
20. Построить validation curve.
21. Построить learning curve.
22. Построить loss curve.
23. Сравнить модели по `f1_macro`.
24. Сохранить лучшую модель.
25. Написать итоговый вывод.

---

# Универсальная короткая болванка для экзамена

Ниже короткий вариант кода, который можно распечатать мелким шрифтом и адаптировать под новый датасет.

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

RANDOM_STATE = 42

DATA_PATH = 'data.csv'
TARGET = 'target'

df = pd.read_csv(DATA_PATH)
print(df.shape)
display(df.head())
df.info()
print(df.isna().sum().sort_values(ascending=False))
print('duplicates:', df.duplicated().sum())
print(df[TARGET].value_counts())

clean_df = df.copy().drop_duplicates()

for col in clean_df.select_dtypes(include=['object', 'category']).columns:
    if col != TARGET:
        clean_df[col] = (
            clean_df[col]
            .astype('string')
            .str.strip()
            .str.lower()
            .replace({'': np.nan, 'nan': np.nan, 'none': np.nan, '?': np.nan})
        )

cols_to_drop = []
for col in ['ID', 'Id', 'id', 'SampleID', 'Unnamed: 0', 'TargetCodeLeak']:
    if col in clean_df.columns:
        cols_to_drop.append(col)
clean_df = clean_df.drop(columns=cols_to_drop)

numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()

def cap_outliers_iqr(data, columns, factor=1.5):
    result = data.copy()
    for col in columns:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        result[col] = result[col].clip(lower=lower, upper=upper)
    return result

clean_df[numeric_cols] = cap_outliers_iqr(clean_df[numeric_cols], numeric_cols)

plt.figure(figsize=(6, 4))
clean_df[TARGET].value_counts().plot(kind='bar')
plt.title('Target distribution')
plt.grid(axis='y')
plt.show()

numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
clean_df[numeric_cols].hist(figsize=(14, 10), bins=20, density=True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
clean_df[numeric_cols].boxplot(rot=45)
plt.grid(axis='y')
plt.show()

corr = clean_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()

X = clean_df.drop(columns=[TARGET])
y_raw = clean_df[TARGET]

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

num_features = X_train.select_dtypes(include=np.number).columns.tolist()
cat_features = X_train.select_dtypes(include=['object', 'category', 'bool', 'string']).columns.tolist()

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features)
])

def evaluate(name, model):
    pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, pred)
    f1 = f1_score(y_valid, pred, average='macro')
    print('\n', '=' * 60)
    print(name)
    print('accuracy:', round(acc, 4))
    print('f1_macro:', round(f1, 4))
    print(classification_report(y_valid, pred, target_names=le.classes_))
    return {'name': name, 'accuracy': acc, 'f1_macro': f1, 'pred': pred}

lr = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))
])

mlp = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=RANDOM_STATE))
])

lr.fit(X_train, y_train)
res_lr = evaluate('Logistic Regression', lr)

mlp.fit(X_train, y_train)
res_mlp = evaluate('MLPClassifier', mlp)

grid_lr = GridSearchCV(
    lr,
    {'model__C': [0.1, 1, 10]},
    cv=3,
    scoring='f1_macro',
    n_jobs=-1
)
grid_lr.fit(X_train, y_train)
res_lr_opt = evaluate('Optimized Logistic Regression', grid_lr.best_estimator_)

grid_mlp = GridSearchCV(
    mlp,
    {
        'model__hidden_layer_sizes': [(32,), (64, 32)],
        'model__alpha': [0.0001, 0.001]
    },
    cv=2,
    scoring='f1_macro',
    n_jobs=1
)
grid_mlp.fit(X_train, y_train)
res_mlp_opt = evaluate('Optimized MLPClassifier', grid_mlp.best_estimator_)

for title, pred in [('LR', res_lr_opt['pred']), ('MLP', res_mlp_opt['pred'])]:
    cm = confusion_matrix(y_valid, pred)
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot()
    plt.title(title)
    plt.show()

param_range = np.array([0.0001, 0.001, 0.01])
train_scores, valid_scores = validation_curve(
    mlp, X_train, y_train,
    param_name='model__alpha',
    param_range=param_range,
    cv=2,
    scoring='f1_macro'
)
plt.semilogx(param_range, train_scores.mean(axis=1), marker='o', label='train')
plt.semilogx(param_range, valid_scores.mean(axis=1), marker='o', label='valid')
plt.legend(); plt.grid(); plt.show()

train_sizes, train_scores, valid_scores = learning_curve(
    grid_mlp.best_estimator_, X_train, y_train,
    cv=2,
    scoring='f1_macro',
    train_sizes=np.linspace(0.3, 1.0, 4)
)
plt.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='train')
plt.plot(train_sizes, valid_scores.mean(axis=1), marker='o', label='valid')
plt.legend(); plt.grid(); plt.show()

best_mlp = grid_mlp.best_estimator_.named_steps['model']
plt.plot(best_mlp.loss_curve_)
plt.title('MLP loss curve')
plt.grid(); plt.show()

results = pd.DataFrame([
    {k: v for k, v in res_lr.items() if k != 'pred'},
    {k: v for k, v in res_lr_opt.items() if k != 'pred'},
    {k: v for k, v in res_mlp.items() if k != 'pred'},
    {k: v for k, v in res_mlp_opt.items() if k != 'pred'},
]).sort_values('f1_macro', ascending=False)

display(results)

if res_mlp_opt['f1_macro'] >= res_lr_opt['f1_macro']:
    best_model = grid_mlp.best_estimator_
else:
    best_model = grid_lr.best_estimator_

joblib.dump(best_model, 'best_exam_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
```

---

# Короткий устный ответ, если попросят объяснить работу

```text
Я решал задачу многоклассовой классификации. Сначала загрузил данные и провёл первичный анализ: посмотрел размер таблицы, типы признаков, пропуски, дубликаты и распределение целевой переменной. Затем выполнил предобработку: удалил дубликаты, очистил грязные категориальные значения, удалил технические признаки и признак-утечку, обработал выбросы методом IQR.

После этого я провёл EDA: построил распределение классов, гистограммы числовых признаков, boxplot, корреляционную матрицу и сравнение признаков по классам. Затем разделил данные на признаки и целевую переменную, закодировал target через LabelEncoder и сделал стратифицированное разбиение на обучающую и валидационную выборки.

Для предобработки использовал Pipeline и ColumnTransformer: числовые признаки заполнялись медианой и масштабировались, категориальные признаки заполнялись самым частым значением и кодировались через OneHotEncoder. Были построены две модели: логистическая регрессия и нейронная сеть MLPClassifier. Для обеих моделей был выполнен подбор гиперпараметров через GridSearchCV.

Качество оценивалось по accuracy, classification_report и основной метрике f1_macro. Дополнительно были построены матрицы ошибок, кривая валидации, кривая обучения и график функции потерь нейронной сети. В конце модели были сравнены по f1_macro, выбрана лучшая и сохранена через joblib.
```

---

# Что распечатать на обратной стороне

Если разрешено печатать код мелким шрифтом, лучше распечатать:

1. Импорты.
2. Проверку данных: `shape`, `head`, `info`, `isna`, `duplicated`, `value_counts`.
3. Очистку категорий.
4. Удаление ID и утечек.
5. IQR-функцию.
6. EDA-графики.
7. `LabelEncoder`.
8. `train_test_split(..., stratify=y)`.
9. `ColumnTransformer`.
10. `LogisticRegression`.
11. `MLPClassifier`.
12. `GridSearchCV`.
13. `classification_report`.
14. `ConfusionMatrixDisplay`.
15. `validation_curve`.
16. `learning_curve`.
17. `loss_curve_`.
18. `joblib.dump`.

Самое ценное для распечатки — универсальная короткая болванка выше.

---

# Финальный смысл решения

Это решение специально сделано не как огромный промышленный ML-проект, а как экзаменационный шаблон:

- код не перегружен;
- критерии закрыты;
- есть классическая модель;
- есть нейронная сеть;
- есть подбор параметров;
- есть графики;
- есть метрики;
- есть сохранение модели;
- есть текстовые выводы;
- можно быстро адаптировать под похожий датасет с 3, 5 или любым другим количеством классов.

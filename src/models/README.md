# Документация по использования PipeLine
---

## Инициализация

Для того чтоы инициализировать моделей для анализа сигалов ЭКГ и генерации промежуточного диагноза, на вход классу PipeLine при инициализации нужно подать набор конфигурационных парметров конфигурационных словарей (signal_anality, generation_config, gen_details, из которых обязателен только первый). Передачу данных конфигурационных словарей можно выполнить одним из трех методов. 

- 1) Задать в качестве параметра config путь до папки в которой лежат все конфигурационные словари (либо в .json, либо в .yaml) 
- 2) Задать в качестве входа путь до одного общего конфигурационного словаря который объядинял бы все остальные
- 3) Задать в качестве входа список путей до кждого конфигурационного словоря по отдельности
- 4) Задать в качестве входа общий конфигурационный файл в формате который использовался во втором методе


### Описание конфигурационных словарей:
---
- 1) Конфигурационный словарь signal_analitic, содержит параметры модели основанной на сверточных слоях, отвечающей за анализ сигналов кфтвщь ЭКГ сигналов и генерацю по ним вектора со значениями RR, PR, QRS, QT. Данный словарь имеет ввид:
```json
{
    "backbone": {
        "projection": {
            "in_channels": [12,32,64],
            "out_channels": [32,64,128],
            "embedding_dim": [32, 32, 64]
        },
        "conv": {
            "kernel_size": [3, 3],
            "padding": 1,
            "stride": 2
        },
        "pool": {
            "kernel_size": [3, 3],
            "padding": 1,
            "stride": 1
        }
    },
    "out_head": {
        "type": "multyoutput",
        "params": {
            "embedding_dim": 16,
            "dims": [
                4
            ]
        }
    }
}
```

#### параметры signal_analitic
- backbone: Dcit
    - .projection: Dict[List[int]] - задает набор размерностей для проектирования которые сверточным слоем основанным на механизме внимания. Кол-во слоев значений данных полей данного параметра рано кол-ву слоев которые используеются моделью.

        - in_channels: List[int] - список входных размерностей для каждого сверточного слоя 
        - out_channels: List[int] - список выходных размерностей для каждого сверточного слоя
        - embedding_dim: List[int] - размерностей для проецироания сжатных представлений активаций сверточных слоев для применения механизма внимания

    - conv: Dict[List[int]] - задает параметры необходимые для работы сверточныйх слоев которые стоят между сверточными слоями основанными на механизме внимания
        - kernel_size: Tuple[int] - размерность ядра свертки для верточных слоев которые использует модельь
        - padding: int - размерность паддинга сверточного слоя
        - stride: int - шаг свертки

    - pool: Dcit[List[int]] - задает параметры необходимые для работы слоев пулинга для стоящийх между всеми вычислительными блоками
        - kernel_size: Tuple[int] - размерность окна по которому проводится пуллинг
        - padding: int - размерность паддинга слоя пуллинга (смысл тот же что и в сверточных слоях)
        - stride: int - шаг пуллинга
    
- out_head: Dict[List[int]] - конфигурация выходной головы модели по анализу входящих сигналов
    - type: srt - тип выходной головы. На данный момент предусмотренно две вариции, одна для классификции, другая для многомерной линейного регрессии
    - params: Dict - набор параметров для того или иного вида модели 
        - [lin. regression model]:
            - embedding_dim: int - размерность проекции лдя предобраобтки входного вектора полученного backbone
            - dims: List[int] - список размерностей выходных голов.
        - [classification model]: 
            - embedding_dim: int - имеет тот же смысл что и для задачи линейной регрессии
            - hiden_featuers: int - размерность скрытых полносвязных слоев модели классификации
            - n_classes: int - кол-во классов для предсказания 

---

- 2) Конфигурационный словарь generation_config содержит в себе параметры контралирующие качество генерируемого языковой моделью текста. Он имеет след. структуру:

```json
{
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": true,
    "num_return_sequences": 1
}
```

#### параметры generation_config

- max_new_tokens: int - длина генерируемой последовательности
- temperature: float, [-1, 1] - коэффициент штрафования за повторяющиеся слова
- top_p: float - вероятностный порог для отобра следующих токенов в ходе генерации текста
- repetition_penalty: float [-1, 1] - аналог штрафа за повторяющиеся токены который можно использовать совместно с temperature
- do_sample: bool - использовать или нет последовательную генерацию токенов
- num_return_sequeces: int - кол-во сгенерированных последовательностей токенов

---

- 3) Конфигурационный словарь gen_details отвечает за доработку входного промпта для модели, в которые затем примешивается нужный вектор со значениями RR, PR, QRS, QT сгенерированными моделью signal_analitic. Он имеет след. стуктуру:

```json
{
    "роль": "Ты помощник медицинского эксперта",
    "задача": """\nТы должен генерировать промежуточные диагнозы на                
                основании диагнозов параметров экг таких как \n                
                RR, PR, QT, QRS, которые ты будешь получать \n                
                в формате текста но в виде list.\n""",
    "результат": """\nТекст с полным описанием экг пациента по \n                
        RR, PR, QT, QRS, дополнительными рекомендациями\n                
        от себя, и предупрежедением о том что ты лишь \n                
        ИИ ассистент и пользователю стоит обратиться за \n                
        более точной консультацией к медработнику. \n                
        Все поля раздели между собой линиями.\n"""
}
```
#### параметры gen_details
В качестве параметров данного конфугурационного словаря можено подовать ключи суммирующие смысл текста описанного в качестве valus к данному ключу. Данные параметры затем читаются объектов PipeLine и используются как дополнение в входному промпту

---

## Пример инициализации класса PipeLine

```python

# config может быть получен одним из методов описанных в пункте инициализации.
config = ...

# путь до .hea, .dot файла с записанными ЭКГ 
path = "<some_path>\\01000_lr"

# инициализируем класс PipeLine передавая ему на вход конфигурационный словарь
pipeline = PipeLine(config=config)

# передаем на вход объекту класса PipeLine в качестве параметра к __call__() методу путь до переменную path
# выходом будет результирующий диагноз с промежуточным диагнозом для пациента по его входной кардиограмме.
out = pipeline(path)
```

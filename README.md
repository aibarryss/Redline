# FinAudit-RL

Воспроизводимая офлайн RL-среда для оценки и потенциального дообучения ИИ-аудиторов (REC GENESIS Hackathon 2026 // FinTech).

## Быстрый старт (единая команда проверки)

Требования: **Python 3.11+**, `tzdata==2025.2` (поддержка IANA-часовых поясов, включая алиас `Asia/Astana`), без GPU, без интернета. Рабочая директория в контейнере: `/app`.

```bash
python3 -m audit_env.verify
# Без перезаписи data/ и trajectories/ в репозитории:
python3 -m audit_env.verify --check-only
```

Команда последовательно выполняет:
1. Генерацию и независимую бухгалтерскую валидацию 10 `train`- и 10 `dev`-эпизодов с разделением на `open/` и `closed/` (`data/train/`, `data/dev/`), а также запись `trajectories/trajectories.jsonl` с проверкой отсутствия утечек эталона и `seed` (при отсутствии прав на запись в каталог проекта автоматически используется временная директория).
2. Прогон 3 не-oracle политик (`random_explorer`, `simple_reconciler`, `flag_all_attack`) в изолированном дочернем процессе ОС (`IsolatedEvaluatorHarness.run_policy_subprocess`) и `oracle` (только на стороне оценщика) на 20 локальных отложенных эпизодах (local holdout-бенчмарк с распределением из раздела 06: 5 чистых, 5 с взаимодействующими ошибками, 10 смешанных; не заменяет секретные seed жюри) + 20 парных проверок на перестановку записей и биективный ремаппинг ID.
3. Полный набор unit-, contract-, security-, matcher- и adversarial-тестов (`tests/`), включая проверки `judge_adapter.py` и примеров `M01–M12`.

> Подробная карта соответствия всем критериям жюри (100 баллов) приведена в [`JUDGES.md`](JUDGES.md).

## Единый адаптер жюри (`judge_adapter.py`)

В корне репозитория (`/app/judge_adapter.py`) реализован тонкий Python-адаптер по спецификации «Определения проверок FinTech»:

* `generate_episode(seed: int, config: dict, output_dir: str) -> None` — обёртка над `audit_env.generator.builder.generate_episode`, сохраняющая в `output_dir` открытые файлы (`manifest.json`, `policy.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv`) и закрытые файлы оценщика (`truth.json`, `provenance.json`).
* `make_env()` — возвращает объект `JudgeEnvAdapter` с методами:
  * `reset(seed=None, config=None) -> (observation, info)` — поддерживает `config={"mode": "closed"|"train", "episode_dir": "<dir>", "team_config": {...}}`. При переданном `episode_dir` загружает ровно этот эпизод без перегенерации и без смены ID; повторный `reset` перезапускает его с нуля. Первое `observation`: `{"task": str, "manifest": object, "policy": object, "remaining_steps": 100}`.
  * `step(action) -> (observation, reward, terminated, truncated, info)` — принимает `{"name": "<действие>", "args": {...}}` для `list_records`, `get_record`, `get_document`, `calculator`, `submit`. Возвращает `{"result": value_or_null, "error": null_or_object, "remaining_steps": int}`. В режиме `closed` поле `reward` всегда равно `0`, а `info` не содержит закрытых данных или диагностики.
* `score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict` — закрытый вызов оценщика:
  * для валидного `submit`: `{"tp": int, "fp": int, "fn": int, "precision": float, "recall": float, "f1": float, "return": float}`;
  * для невалидного `submit`: `{"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1}`.

### Допустимые таблицы, схема результатов и соответствие типов ошибок

* **Таблицы адаптера:** `ledger` (ключ `row_id`), `invoices` (ключ `invoice_id`, документ `doc_id`), `payments` (ключ `payment_id`, документ `doc_id`), `fx` (ключ `fx_id = FX-<date>-<source>-<target>`).
* **Представление результатов `step`:**
  * `list_records(table, cursor, limit)` → `{"records": [object], "next_cursor": null_or_string}` (первый `cursor = null`, `1 <= limit <= 50`);
  * `get_record(table, id)` → объект запрошенной записи;
  * `get_document(doc_id)` → объект запрошенного документа (`invoice` или `payment`);
  * `calculator(expression)` → десятичная строка (`Decimal`, например `"0.3"`);
  * `submit(findings)` → `{"accepted": true}` либо ограниченная ошибка структуры `{"code": "INVALID_FINDINGS_SCHEMA", "message": ...}`.
* **Соответствие 4 базовых типов ошибок (§02):**
  1. Дубли счетов → `duplicate_invoice`
  2. Пропущенное или частичное распределение платежа → `payment_allocation_mismatch`
  3. Неверная дата или курс конвертации → `fx_rate_or_date_error`
  4. Ошибка отнесения к отчётному периоду → `cutoff_period_error`
* **Два взаимодействующих сценария (`interacting_scenarios` в `EpisodeConfig`):** `interacting_fx_partial_cutoff`, `interacting_dup_fx_allocation`.
* **Нормализация matcher:** ID (`affected_ids`, `evidence_ids`) и `currency` сравниваются точно с учётом регистра и пробелов; неизвестный строковый `type` считается несопоставимым (`FP`), а не ошибкой структуры; при нескольких максимальных паросочетаниях выбирается лексикографически наименьшая последовательность `(issue_id, finding_index)`. Допуск суммы: τ=0 по умолчанию, публикуется в `manifest.amount_tolerance_minor` (и `policy.json`).

## Наглядное демо работы среды

```bash
python3 demo.py --seed 2026
```

Скрипт запускает `SimpleReconcilerPolicy` (и остальные не-oracle политики в сводной таблице) в отдельном процессе воркера (`IsolatedEvaluatorHarness.run_policy_subprocess`, `mode=eval`), печатает каждый шаг расследования, скрытый отчёт оценщика и сравнительную таблицу всех 4 политик на одном эпизоде.

## Запуск в Docker (офлайн)

```bash
docker build -t finaudit-rl .
docker run --rm --network none --cpus 2 --memory 4g finaudit-rl
```

## Запуск только тестов

```bash
python3 -m unittest discover -s tests -v
```

## Структура репозитория

- `judge_adapter.py` — единый адаптер жюри (`generate_episode`, `make_env`, `score_episode`).
- `audit_env/constants.py`, `audit_env/schemas.py` — контракты данных, лимиты, строгая валидация конфигов и `findings.json`, хранение эпизодов в форматах `split` (`open/` + `closed/`) и `flat`.
- `audit_env/generator/` — детерминированный генератор чистой сбалансированной компании (`company.py`), 4 базовых и 2 взаимодействующих сценариев ошибок (`mutations.py`), сборщик (`builder.py`) и слой перестановок/ремаппинга ID (`transforms.py`).
- `audit_env/validators/consistency.py` — независимый валидатор балансов, границ периода, расхождений, выводимости FX-даты и достаточности доказательств.
- `audit_env/env/` — ядро `AuditEnv` (`engine.py`), безопасный `Decimal`-калькулятор с ограниченным контекстом (`calculator.py`), двухпроцессная изоляция оценщика и воркера политик (`isolated_runner.py`, `policy_worker.py`).
- `audit_env/scoring/` — открытый двудольный matcher с лексикографическим тай-брейкингом (`matcher.py`) и формула метрик/награды (`reward.py`).
- `audit_env/policies/` — 4 базовые политики (`random_explorer`, `simple_reconciler`, `flag_all_attack`, `oracle`).
- `demo.py` — пошаговая CLI-демонстрация расследования и сравнения 4 политик.
- `data/train/`, `data/dev/` — 10 открытых `train`- и 10 `dev`-эпизодов (разделены на `open/` и `closed/`).
- `trajectories/trajectories.jsonl` — записанные траектории базовых политик без закрытого эталона в наблюдениях.
- `DESIGN.md`, `JUDGES.md` — записка о дизайне, измеренный разбор сбоев, честные ограничения и карта проверок для жюри.

## Измеренные результаты на 20 локальных отложенных эпизодах (two-process isolated)

> Это воспроизводимый локальный holdout-бенчмарк с распределением из раздела 06 ТЗ (seed открыты в `verify.py`); секретные seed и параметры жюри подаются отдельно при закрытой проверке.

| Политика | Precision | Recall | F1 | Mean Return | Mean Steps | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `random_explorer` (isolated subprocess) | 0.0000 | 0.0000 | 0.1000 | -0.8016 | 9.00 | 1.128 |
| `simple_reconciler` (isolated subprocess) | 0.7500 | 0.7500 | 1.0000 | 0.9660 | 18.00 | 1.232 |
| `flag_all_attack` (isolated subprocess) | 0.0000 | 0.0000 | 0.0000 | -1.0000 | 3.00 | 1.095 |
| `oracle` (evaluator-only) | 0.7500 | 0.7500 | 1.0000 | 1.0000 | 1.00 | 0.063 |

> Примечание: по правилам раздела 05 на 5 чистых эпизодах при пустом эталоне и пустом ответе `P = 0.0`, `R = 0.0`, `F1 = 1.0`, поэтому среднее `P` и `R` по 20 эпизодам для идеального решения составляет `15 / 20 = 0.7500`, а `F1 = 1.0000`.

## Раскрытие использованных библиотек и инструментов (§08 Положения)

- **Среда исполнения и зависимости:** стандартная библиотека Python 3.11 + пакет `tzdata==2025.2` (база часовых поясов IANA); официальный базовый образ `python:3.11-slim`. Внешние API, GPU и интернет не используются.
- **AI-инструменты:** при разработке, ревью кода и проверке пограничных случаев использовались AI-ассистенты для генерации шаблонов, аудита безопасности и написания тестов; вся архитектура, бухгалтерская модель, логика проверок и финальная верификация выполнены и проверены командой.

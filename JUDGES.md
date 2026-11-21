# Карта проверки для жюри (`JUDGES.md`)

| Критерий ТЗ (раздел 10) | Проверка | Файл | Команда запуска |
| :--- | :--- | :--- | :--- |
| Подключение жюри (Определения проверок) | Единый адаптер `generate_episode`, `make_env` (`reset`/`step` с envelope `{"name", "args"}` и `mode="closed"`/`"train"`), `score_episode` (включая `null`-метрики при невалидном `submit`), примеры `M01–M12` и контрактные тесты `C01–C20` | `judge_adapter.py`, `tests/test_judge_adapter_and_spec.py` | `python3 -m unittest tests.test_judge_adapter_and_spec -v` |
| Контракт и изоляция (25 б.) | Контракт `reset`/`step`, пагинация (≤50), бюджет (≤100), тайм-аут, AST-калькулятор, повторный `submit`, двухпроцессная изоляция и защита от Path Traversal | `tests/test_contract_isolation.py` | `python3 -m unittest tests.test_contract_isolation -v` |
| Контракт и изоляция (25 б.) | Фаззинг утечек в `mode="eval"` на 50 эпизодах для 4 политик и мусорных действий (`None`, `int`, `list`, лишние поля, строки 100 КБ, неверные `table`/`cursor`/`limit`/`id`) | `tests/test_leak_fuzz.py` | `python3 -m unittest tests.test_leak_fuzz -v` |
| Данные и эталон (25 б.) | Независимый валидатор балансов, 4 базовых типа ошибок, 2 взаимодействующих сценария, чистые отвлекающие кейсы, отсутствие утечек в траекториях | `tests/test_data_truth.py`, `audit_env/validators/consistency.py` | `python3 -m unittest tests.test_data_truth -v` |
| Данные и эталон (25 б.) | Детерминированный фаззинг 200 случайных конфигураций (валюты вкл. `JPY`, часовые пояса, оба FX-правила, оба округления, 0–3 ошибок каждого типа, оба интерактивных сценария): валидатор, `oracle F1 == 1.0`, `simple_reconciler` без исключений | `audit_env/fuzz.py` | `python3 -m audit_env.fuzz --n 200` |
| Надёжность награды (25 б.) | Инварианты награды, двудольный matcher с точным лексикографическим тай-брейкингом, допуск по сумме, ошибки схемы (`-1.0`), таблица атак (угаданные ID без доказательств, чужая валюта, повтор 30 раз, пустые `evidence_ids`, все 4 типа на каждый ID, `flag_all`) | `tests/test_reward_attacks.py`, `audit_env/scoring/matcher.py`, `audit_env/scoring/reward.py` | `python3 -m unittest tests.test_reward_attacks -v` |
| Воспроизводимость и обобщение (15 б.) | Битовая воспроизводимость по `(seed, config, actions)` и 20 парных проверок инвариантности при перестановке строк и биективном ремаппинге ID | `tests/test_reproducibility_invariance.py`, `audit_env/generator/transforms.py` | `python3 -m unittest tests.test_reproducibility_invariance -v` |
| Инженерия и объяснение (10 б.) | Единая офлайн-команда верификации (генерация `data/train` + `data/dev` + `trajectories.jsonl`, двухпроцессный бенчмарк 3 политик + evaluator-only oracle на 20 локальных отложенных эпизодах, прогон всех тестов) | `audit_env/verify.py` | `python3 -m audit_env.verify` |
| Инженерия и объяснение (10 б.) | Офлайн-воспроизведение в Docker без сети (Python 3.11, 2 CPU, 4 ГБ RAM) и записка с измеренным разбором сбоев и ограничений | `Dockerfile`, `DESIGN.md` | `docker build -t finaudit-rl . && docker run --rm --network none --cpus 2 --memory 4g finaudit-rl` |

## Справочник для независимой проверки жюри (Блоки 2 и 4)

### 1. Примеры `team_config` (`EpisodeConfig`) под матрицу `D01–D20`

Все ключи опциональны (дефолты: `amount_tolerance_minor=0`, `base_currency="USD"`, `timezone="Asia/Almaty"`, `fx_rate_date_rule="invoice_issue_date"`, `rounding_mode="ROUND_HALF_UP"`):

- **Чистый эпизод (`D01–D05`, `truth.issues == []`):**
  ```json
  {"error_counts": {"duplicate_invoice": 0, "payment_allocation_mismatch": 0, "fx_rate_or_date_error": 0, "cutoff_period_error": 0}, "num_split_payment_distractors": 3, "num_refund_distractors": 2, "num_recurring_distractors": 2}
  ```
- **Один изолированный тип ошибки (`D06–D15`):** задать нужный счётчик `>= 1` в `error_counts` (остальные `0`), например `{"error_counts": {"fx_rate_or_date_error": 1}, "fx_rate_date_rule": "payment_date"}`.
- **Интерактивные сценарии (`D16–D20`):**
  - Сценарий 1: `{"error_counts": {}, "interacting_scenarios": ["interacting_fx_partial_cutoff"]}`
  - Сценарий 2: `{"error_counts": {}, "interacting_scenarios": ["interacting_dup_fx_allocation"]}`

### 2. План счетов и структура `provenance.json`

- **Счета в `ledger.csv`:** `1010_CASH`, `1200_ACCOUNTS_RECEIVABLE`, `2000_ACCOUNTS_PAYABLE`, `2150_UNAPPLIED_CASH`, `4000_REVENUE`, `4100_SALES_RETURNS`, `5000_OPERATING_EXPENSE`. До мутаций для каждой операции и всей книги выполняется `sum(debit_minor) == sum(credit_minor)`.
- **Поля `provenance.json`:** `episode_id`, `config_version`, `clean_balance_check` (бухгалтерский баланс чистой книги до мутаций, включая `clean_ledger_rows`), `distractors`, `mutations` (список объектов с ключами `issue_id`, `type`, `before`, `after`, `causal_note` и метаданными мутации), `transform_applied`.

### 3. Пространства ID для парных проверок (`P01–P20`)

- **Заменяемые surrogate ID:** `row_id` (`R-*`), `doc_id` (`DI-*`, `DP-*`), `invoice_id` (`I-*`), `payment_id` (`P-*`), `entity_id` (`E-*`), а также все ссылки на них в `ledger.csv`, `payments.jsonl` (`invoice_refs`), `truth.json` (`affected_ids`, `acceptable_evidence_sets`) и `provenance.json`.
- **Не surrogate ID:** ключ FX `fx_id = FX-<date>-<source_currency>-<target_currency>` формируется из даты и валютной пары и не переименовывается при смене ID.

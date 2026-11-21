# Карта проверки для жюри (`JUDGES.md`)

## Чеклист обязательных материалов сдачи (§09)

| Материал (§09) | Путь в репозитории |
| :--- | :--- |
| Исходный код, лицензия, зависимости | `audit_env/`, `LICENSE`, `requirements.txt`, `pyproject.toml` |
| README и инструкции контейнера | `README.md`, `Dockerfile`, `.dockerignore` |
| Matcher, награда, схемы, адаптер | `audit_env/scoring/`, `audit_env/schemas.py`, `judge_adapter.py` |
| 10 `train`- и 10 `dev`-эпизодов | `data/train/train_01..10/`, `data/dev/dev_01..10/` (`open/` + `closed/`) |
| Тесты среды, сопоставления и награды | `tests/` (43 теста, включая `C01–C20`, `M01–M12`, `test_leak_fuzz.py`) |
| Четыре адаптера базовых политик | `audit_env/policies/` (`random_explorer`, `simple_reconciler`, `flag_all_attack`, `oracle`) |
| Записанные траектории (0 утечек) | `trajectories/trajectories.jsonl` (300 шагов) |
| Записка о дизайне | `DESIGN.md` |
| Демо работы среды | `demo.py` (`python3 demo.py --seed 2026`) |

## Карта соответствия критериям оценки (§10 — 100 баллов)

| Критерий ТЗ (раздел 10) | Проверка | Файл | Команда запуска |
| :--- | :--- | :--- | :--- |
| Подключение жюри (Определения проверок) | Единый адаптер `generate_episode`, `make_env` (`reset`/`step` с envelope `{"name", "args"}` и `mode="closed"`/`"train"`), `score_episode` (включая `null`-метрики при невалидном `submit`), примеры `M01–M12` и контрактные тесты `C01–C20` | `judge_adapter.py`, `tests/test_judge_adapter_and_spec.py` | `python3 -m unittest tests.test_judge_adapter_and_spec -v` |
| Контракт и изоляция (25 б.) | Контракт `reset`/`step`, пагинация (≤50), бюджет (≤100), тайм-аут, AST-калькулятор, повторный `submit`, двухпроцессная изоляция и защита от Path Traversal | `tests/test_contract_isolation.py` | `python3 -m unittest tests.test_contract_isolation -v` |
| Контракт и изоляция (25 б.) | Фаззинг утечек в `mode="eval"` на 50 эпизодах для 4 политик и мусорных действий (`None`, `int`, `list`, лишние поля, строки 100 КБ, неверные `table`/`cursor`/`limit`/`id`) | `tests/test_leak_fuzz.py` | `python3 -m unittest tests.test_leak_fuzz -v` |
| Данные и эталон (25 б.) | Независимый валидатор балансов, 4 базовых типа ошибок, 2 взаимодействующих сценария, чистые отвлекающие кейсы, отсутствие утечек в траекториях | `tests/test_data_truth.py`, `audit_env/validators/consistency.py` | `python3 -m unittest tests.test_data_truth -v` |
| Данные и эталон (25 б.) | Детерминированный фаззинг 200 случайных конфигураций (валюты вкл. `JPY`, часовые пояса, оба FX-правила, оба округления, 0–3 ошибок каждого типа, оба интерактивных сценария): валидатор, `oracle F1 == 1.0`, `simple_reconciler` без исключений | `audit_env/fuzz.py` | `python3 -m audit_env.fuzz --n 200` |
| Надёжность награды (25 б.) | Инварианты награды, двудольный matcher с точным лексикографическим тай-брейкингом, допуск по сумме (`τ=0` по умолчанию), ошибки схемы (`-1.0`), таблица атак | `tests/test_reward_attacks.py`, `audit_env/scoring/matcher.py`, `audit_env/scoring/reward.py` | `python3 -m unittest tests.test_reward_attacks -v` |
| Воспроизводимость и обобщение (15 б.) | Битовая воспроизводимость по `(seed, config, actions)` и 20 парных проверок инвариантности при перестановке строк и биективном ремаппинге ID | `tests/test_reproducibility_invariance.py`, `audit_env/generator/transforms.py` | `python3 -m unittest tests.test_reproducibility_invariance -v` |
| Инженерия и объяснение (10 б.) | Единая офлайн-команда верификации (генерация `data/train` + `data/dev` + `trajectories.jsonl`, двухпроцессный бенчмарк 3 политик + evaluator-only oracle на 20 локальных отложенных эпизодах, прогон всех тестов) | `audit_env/verify.py` | `python3 -m audit_env.verify` |
| Инженерия и объяснение (10 б.) | Офлайн-воспроизведение в Docker без сети (Python 3.11, 2 CPU, 4 ГБ RAM) и записка с измеренным разбором сбоев и ограничений | `Dockerfile`, `DESIGN.md` | `docker build -t finaudit-rl . && docker run --rm --network none --cpus 2 --memory 4g finaudit-rl` |

## Быстрый smoke-тест `judge_adapter.py` (copy-paste)

```bash
python3 -c '
import tempfile, judge_adapter as ja
with tempfile.TemporaryDirectory() as d:
    ja.generate_episode(seed=2026, config={}, output_dir=d)
    env = ja.make_env()
    obs, info = env.reset(config={"mode": "closed", "episode_dir": d})
    o1, r1, term1, trunc1, _ = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 5}})
    findings = {"episode_id": obs["manifest"]["episode_id"], "findings": []}
    o2, r2, term2, trunc2, _ = env.step({"name": "submit", "args": {"findings": findings}})
    score = ja.score_episode(d, findings, inspection_steps=1)
    assert r1 == 0 and r2 == 0 and term2 and not trunc2 and o2["result"] == {"accepted": True}
    print("SMOKE OK:", {"records_page": len(o1["result"]["records"]), "score": score})
'
```

## Справочник для независимой проверки жюри (Блоки 2, 3 и 4)

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

### 2. Канонические правила `affected_ids` и `acceptable_evidence_sets` по типам ошибок

| Тип ошибки | Канонические `affected_ids` (множество) | Допустимый набор доказательств (`acceptable_evidence_sets`) |
| :--- | :--- | :--- |
| `duplicate_invoice` | `{orig_invoice_id, dup_invoice_id}` (`I-*`, `I-*`) | `{orig_doc_id, dup_doc_id}` (`DI-*`, `DI-*`) + хотя бы один `row_id` проводки дубля (`R-*`) |
| `payment_allocation_mismatch` | `unapplied`: `{payment_id}` (`P-*`); `partial_mismatch`: `{invoice_id, payment_id}` (`I-*`, `P-*`) | `unapplied`: `{payment_doc_id}` + ≥1 `row_id` платежа; `partial_mismatch`: `{invoice_doc_id, payment_doc_id}` + ≥1 `row_id` платежа |
| `fx_rate_or_date_error` | `{invoice_id, ledger_row_1, ledger_row_2}` (`I-*` и обе проводки счёта `R-*`) | `{invoice_doc_id, fx_id_по_policy}` (`DI-*`, `FX-<date>-<src>-<dst>`) + ≥1 `row_id` проводки счёта |
| `cutoff_period_error` | `{invoice_id или payment_id, ledger_row_1, ledger_row_2}` (`I-*`/`P-*` и обе проводки `R-*`) | `{doc_id}` (`DI-*` или `DP-*`) + ≥1 `row_id` проводки (`R-*`) |

### 3. План счетов и структура `provenance.json`

- **Счета в `ledger.csv`:** `1010_CASH`, `1200_ACCOUNTS_RECEIVABLE`, `2000_ACCOUNTS_PAYABLE`, `2150_UNAPPLIED_CASH`, `4000_REVENUE`, `4100_SALES_RETURNS`, `5000_OPERATING_EXPENSE`. До мутаций для каждой операции и всей книги выполняется `sum(debit_minor) == sum(credit_minor)`.
- **Поля `provenance.json`:** `episode_id`, `config_version`, `clean_balance_check` (бухгалтерский баланс чистой книги до мутаций, включая `clean_ledger_rows`), `distractors`, `mutations` (список объектов с ключами `issue_id`, `type`, `before`, `after`, `causal_note` и метаданными мутации), `transform_applied`.

### 4. Пространства ID для парных проверок (`P01–P20`)

- **Заменяемые surrogate ID:** `row_id` (`R-*`), `doc_id` (`DI-*`, `DP-*`), `invoice_id` (`I-*`), `payment_id` (`P-*`), `entity_id` (`E-*`), а также все ссылки на них в `ledger.csv`, `payments.jsonl` (`invoice_refs`), `truth.json` (`affected_ids`, `acceptable_evidence_sets`) и `provenance.json`.
- **Не surrogate ID:** ключ FX `fx_id = FX-<date>-<source_currency>-<target_currency>` формируется из даты и валютной пары и не переименовывается при смене ID.

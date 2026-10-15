# Overview

This repository implements a synthetic Instamart Customer Intelligence & Support Copilot in four connected parts: Part A cleans deliberately dirty customer, product, and order exports and stores trusted tables in SQLite; Part B builds RFM segments and a leakage-controlled lapse model; Part C redacts personal information and triages support tickets with the supplied offline simulator; Part D retrieves policy passages and runs a checkpointed refund agent. All data is fictional. The supplied `generate_data.py` and `mock_llm.py` were recovered from the embedded source supplied with the assignment reference; their recovered contents are kept unchanged.

# Reproduce

Run from the repository root with Python 3.11+ and SQLite 3.25+. `requirements.txt` pins the four direct third-party dependencies; pip resolves their compatible transitive dependencies. On Windows PowerShell:

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python run_all.py`

On macOS/Linux, use `source .venv/bin/activate` in step 2. The generator's final line is verified as `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. `run_all.py` executes the Part A, B, C, and D pipelines in order, creates the required outputs, and prints the Part A validation checkpoint. To run the acceptance tests after reproduction: `python -m unittest discover -s tests -v`.

# Part A

## Raw audit

Raw shapes and pandas dtypes:

| Table | Shape | Columns and dtypes |
|---|---:|---|
| customers | 1200 × 5 | customer_id str; city str; signup_date str; age float64; gender str |
| products | 40 × 4 | product_id str; product_name str; category str; price_inr int64 |
| orders | 13084 × 9 | order_id str; customer_id str; product_id str; quantity int64; order_ts str; status str; discount_pct int64; order_amount str; delivery_minutes float64 |
| tickets | 164 × 5 | ticket_id str; customer_id str; order_id str; text str; true_category str |

Raw null counts and percentages (all unlisted columns have zero missing values):

| Table.column | Missing | Percent |
|---|---:|---:|
| customers.age | 61 | 5.0833% |
| customers.gender | 30 | 2.5000% |
| orders.order_amount | 53 | 0.4051% |
| orders.delivery_minutes | 2541 | 19.4207% |
| all products columns | 0 | 0% |
| all remaining customer, order, and ticket columns | 0 | 0% |

`tickets.csv` is read with `keep_default_na=False`; its four empty ticket texts are intentionally empty strings, not NA values. The raw amount audit separately counts values that fail numeric conversion, including both blank/NA and malformed values.

| Raw audit checkpoint | Measured |
|---|---:|
| Order rows / exact duplicate rows / pending rows / distinct order IDs | 13084 / 119 / 656 / 12309 |
| Non-numeric order amounts / negative delivery times / blank delivery times | 120 / 106 / 2541 |
| Blank ages / ages outside 10–100 / raw city spellings | 61 / 6 / 24 |
| Nonblank raw gender spellings / blank genders | 10 / 30 |

## Cleaning and storage

Customer cities were trimmed and title-cased, yielding six supported cities. Gender variants map to `male`, `female`, or `unknown`. The 61 blank and six out-of-range ages (67 total) were filled with each city's median using `groupby` and `transform`; signup dates were parsed as datetimes. Final city counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts: female 610, male 560, unknown 30. Ages are complete and range from 18 to 60.

Order cleaning follows the required sequence: parse timestamps; remove 119 exact duplicate rows; keep the latest timestamp per order ID (resolving earlier pending records); coerce amount values to numeric and reconstruct 112 missing amounts using product price × quantity × (1 − discount/100), rounded to two decimals; turn 105 negative delivery values remaining after deduplication into missing values; fill 774 missing completed-order delivery times with the completed-order median for that product category; leave cancelled delivery times missing. The final table has 12309 unique orders: 11110 completed and 1199 cancelled, no pending rows, no missing amounts, no missing completed delivery times, and 1199 missing cancelled delivery times. Completed-order revenue is INR 2,998,668.80.

The clean CSVs are `data/clean/customers.csv`, `data/clean/products.csv`, and `data/clean/orders.csv`. `data/instamart.db` has `customers`, `products`, and `orders` tables; `order_ts` is stored as `YYYY-MM-DD HH:MM:SS`. The validator asserts multiple customer, order, revenue, and query checkpoints and prints `ALL CHECKPOINTS PASSED` only when they all hold.

## SQL and pandas analysis

`sql/queries.sql` contains seven queries, each executed with `pd.read_sql_query`:

| Query | Result |
|---|---|
| Q1 — completed orders above INR 500, grouped and filtered above INR 8000 | 30 customers |
| Q2 — completed revenue by city and category | 30 cells; INR 2,998,668.80 total; largest is Bengaluru / Electronics at INR 233,909.45 |
| Q3 — products priced above catalogue average | 11 products |
| Q4 — customers with no completed order | 25 customers |
| Q5 — customers spending above the mean among customers with completed spend | 550 customers; mean spend INR 2,552.06 |
| Q6 — city-partitioned spend rankings | Top customer per city: Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10. Eight rows have `RANK()` different from `ROW_NUMBER()` because tied spend shares a rank while row numbers remain unique. |
| Q7 — daily revenue and running total | 183 days; final cumulative revenue INR 2,998,668.80 |

The pandas Q2 pivot table grand total is INR 2,998,668.80, matching SQL. City totals:

| City | Completed spend (INR) | Mean order (INR) | Orders |
|---|---:|---:|---:|
| Bengaluru | 523643.40 | 276.47 | 1894 |
| Chennai | 463629.15 | 272.72 | 1700 |
| Delhi | 502829.80 | 243.62 | 2064 |
| Hyderabad | 528589.90 | 284.95 | 1855 |
| Mumbai | 479058.50 | 297.00 | 1613 |
| Pune | 500918.05 | 252.48 | 1984 |

Completed order amounts have mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00, lower fence INR −283.50, upper fence INR 645.30, zero lower outliers, and 897 upper outliers. The correlation matrix is:

|  | order_amount | discount_pct | quantity | delivery_minutes |
|---|---:|---:|---:|---:|
| order_amount | 1.000 | −0.095 | 0.087 | −0.003 |
| discount_pct | −0.095 | 1.000 | −0.000 | −0.010 |
| quantity | 0.087 | −0.000 | 1.000 | −0.013 |
| delivery_minutes | −0.003 | −0.010 | −0.013 | 1.000 |

The data describes synthetic orders across six cities and five product categories. Amounts are strongly right-skewed, with 897 high-side IQR outliers; raw exports also contain duplicates, malformed amounts, missing values, and impossible delivery durations. Order amount and discount percentage have the largest absolute correlation (−0.095), followed by amount and quantity (+0.087). These relationships are small and do not establish causation.

# Part B

## RFM, clusters, and PCA

RFM uses completed orders and the fixed snapshot `2025-10-01 00:00:00`. The 1175 customers with completed orders have median recency 19 days (range 0–180), mean frequency 9.46 (maximum 33), mean monetary value INR 2,552.06 (maximum INR 14,766.55). RFM features are standardized before K-means.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.76 | 0.465 |
| 3 | 1268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 was selected: it has the best measured silhouette, a clear reduction in inertia, and interpretable customer profiles. K=8 necessarily has lower inertia than K=2 because more centroids can only reduce within-cluster squared distances; lower inertia by itself does not mean more useful segments.

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Suggested action |
|---|---:|---:|---:|---:|---|
| Developing | 436 | 28.86 | 3.77 | 1057.62 | Encourage a second/third purchase with relevant replenishment offers. |
| Premium | 138 | 24.12 | 6.77 | 7129.70 | Protect with high-touch support and early access to relevant offers. |
| Frequent | 419 | 8.77 | 19.40 | 3410.02 | Reinforce habit with convenient repeat-order suggestions. |
| At-Risk | 182 | 109.04 | 2.20 | 685.99 | Test a targeted win-back offer and monitor response. |

Premium is assigned to the highest-mean-monetary cluster. Cluster IDs are arbitrary labels, not a ranking. PCA on standardized RFM explains 0.634 and 0.218 of variance (0.852 combined). The segment output is `outputs/customer_segments.csv`.

## Lapse prediction

The pre-September cohort has 1156 customers; 443 (38.32%) are labelled lapsed. Features use completed orders strictly before `2025-09-01`, plus pre-cutoff cancellation rate, average discount, average delivery, favourite category, city, age, gender, and signup tenure. Segment labels are excluded. A stratified `random_state=42` split yields 924 training and 232 test rows. The `ColumnTransformer` fits numeric scaling and `OneHotEncoder(drop="first")` only on training data; all feature values are nonmissing. Learned training frequency mean is 7.307 versus full-cohort 8.029 because the scaler is fitted only on the training split.

| Model | Search result | Best CV F1 | Test accuracy | Lapsed precision | Lapsed recall | Lapsed F1 | Confusion matrix (actual rows: stay, lapse; predicted columns: stay, lapse) |
|---|---|---:|---:|---:|---:|---:|---|
| Majority baseline | all stay | — | 0.616 | 0.000 | 0.000 | 0.000 | `[[143, 0], [89, 0]]` |
| Logistic Regression | C=100 | 0.651 | 0.728 | 0.644 | 0.652 | 0.648 | `[[111, 32], [31, 58]]` |
| Random Forest | depth=8, leaf=10 | 0.687 | 0.720 | 0.633 | 0.640 | 0.637 | `[[110, 33], [32, 57]]` |

Logistic Regression was selected on the held-out test F1 (0.648 versus 0.637), exceeding the required 0.55 threshold. Missing a real lapse loses a retention opportunity; false positives spend outreach capacity and may annoy loyal customers, so the final threshold should reflect the business costs. Both models use five-fold `GridSearchCV`; the forest grid covers 45 fits. The measured best Logistic Regression C differs from the brief's approximate C=1 reference under the installed library versions; the measured model still satisfies the required test F1.

Oversampling duplicated lapsed training rows only, with seed 42, yielding 570 examples per class. On the unchanged test set, Logistic Regression precision/recall/F1 changed from 0.644/0.652/0.648 to 0.600/0.843/0.701. Recall moved most, identifying more lapsed customers at the cost of more false-positive contacts.

The five largest Random Forest importances were monetary 0.280, frequency 0.278, recency 0.109, average delivery 0.060, and cancellation rate 0.052. The three largest-magnitude Logistic Regression coefficients were frequency −1.697, favourite category Electronics −1.135, and favourite category Grocery −0.449; conditional on the other features, these are associated with lower lapse odds than the encoder's reference levels, not causal effects. The selected pipeline is saved as `outputs/lapse_model.joblib`; reloaded predictions matched exactly. `outputs/retention_list.csv` contains 10 Premium test customers ranked by predicted lapse probability; four actually lapsed.

**Premium-label limitation:** The Premium flag is a post-hoc whole-window label derived from data that includes the September lapse-label period; therefore, the retention list is illustrative and is not a leakage-free evaluation. The model's own features still use only pre-cutoff orders.

# Part C

Tickets are processed in original file order. Four blank tickets are `skipped_empty`; four tickets longer than 400 estimated tokens are `flagged_long` and are not sent to the model (each long example estimates to 734 tokens). Email, phone, and order-ID redaction was applied in that order: 55 emails, 91 phones, and 132 order IDs (278 total substitutions); post-redaction checks found zero remaining matches. The user message contains only the redacted ticket text, and code asserts no raw match reaches a prompt.

Both prompts require JSON with one of `billing`, `delivery`, `technical`, or `fraud`. The few-shot prompt adds two labelled examples, including fraud; the zero-shot prompt has no `EXAMPLES:` section. Every independent run resets the simulator. Each full run had 19 transient exceptions, all recovered with a jittered exponential retry; every retry delay is logged in `outputs/triage_retries.log`. Output parsing accepts valid allowed-category JSON as `classified_json`, exactly one allowed category extracted from malformed text as `classified_keyword`, and otherwise flags for human review without guessing.

| Run | Classified | Correct | Accuracy | Total tokens | Tokens / classified |
|---|---:|---:|---:|---:|---:|
| Zero-shot, no budget | 147 | 126 | 85.71% | 9088 | 61.82 |
| Few-shot, no budget | 147 | 147 | 100.00% | 14080 | 95.78 |

The zero-shot confusion table:

| Actual \ predicted | billing | delivery | fraud | technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 11 | 10 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 0 | 0 | 23 | 0 |
| technical | 0 | 0 | 0 | 29 |

Few-shot used 4992 additional tokens (about 55% more) and improved measured accuracy by 14.29 percentage points. That cost is warranted when category accuracy and safe fraud routing matter more than token economy; a lower-stakes or tightly budgeted queue may prefer zero-shot with human review for uncertain outputs.

With a zero-shot demo budget of 6000 tokens, usage reached 6018 across calls, and 53 tickets were marked `budget_exhausted` before any further model calls. The final few-shot run uses a budget of 100000 and writes all 164 rows to `outputs/triage_results.csv`; `outputs/triage_budget_demo.csv` records the budget demonstration. Routing prioritizes human-review statuses, then fraud to the high-priority security team, then Premium customers to the high-priority premium desk, and all remaining tickets to the normal support queue.

# Part D

## Policy retrieval and grounding

Each policy file is split into 40-word chunks with 10-word overlap (step 30): refund 4, delivery 3, payment 3, escalation 3, privacy 2; 15 total. `TfidfVectorizer` plus cosine similarity retrieves the top `k=3`; all three results for `refund policy late delivery` come from `refund_policy.txt`. TF counts term frequency in a chunk; IDF downweights terms common across chunks. TF-IDF is lexical and can miss equivalent meanings expressed with different words.

The grounded assistant sends a `POLICY ASSISTANT` system message and a user message with the three source-prefixed chunks on separate context lines followed by the question. On all 14 evaluation questions, the overlapping retriever returned the expected source for 12/12 answerable questions; answers contained the expected fact for 12/12, and both unsupported questions received the required “I could not find” fallback. With 40-word chunks and zero overlap, source hits remained 12/12, expected facts fell to 10/12, and unsupported fallbacks remained 2/2. The refund-delivery-time question and failed-payment/48-hour question failed because their facts cross a chunk boundary without overlap.

## Tools and agent evaluation

The `TOOLS_SPEC` and `TOOL_MAP` provide five validated tools: `get_order` reads a single row from SQLite; `search_policy` returns retrieved passages; `calculate_refund` returns the full order amount only above 45 delivery minutes; `issue_refund` records eligible refunds up to INR 1000; `escalate_to_human` records eligible amounts above INR 1000 for approval. Refund actions recheck completion status, delivery eligibility, amount, and the approval threshold against the database. The JSONL ledger is a simulation only and does not move money.

| Scenario | Status | Refund (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 907 |
| S2 | REFUND | 135.00 | 5 | 907 |
| S3 | ESCALATE | 1999.00 | 5 | 894 |
| S4 | DENIED | 0.00 | 4 | 613 |
| S5 | DENIED | 0.00 | 3 | 343 |
| S6 | NOT_FOUND | 0.00 | 2 | 76 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3012 |

`outputs/agent_runs.jsonl` records each scenario's result and all tool steps; `outputs/refund_ledger.jsonl` records simulated issue/escalation events with UTC timestamps. `state/` stores full conversation checkpoints after each tool observation. S1's crash simulation stopped after the second checkpointed tool and resumed to the same INR 128.25 result. Across the initial run and resume, exactly these tools executed once each: `get_order` → `search_policy` → `calculate_refund` → `issue_refund`.

Safety demonstrations: the S1 token-limit run stopped with `FALLBACK_TOKEN_LIMIT` after 3 model calls and 360 accumulated tokens (limit 200), without executing the tool requested by the over-limit response. A zero-second time limit returned `FALLBACK_TIMEOUT` before its first model call (0 calls, 0 tokens). S7 made all six allowed model calls before returning `FALLBACK_MAX_ITERATIONS`.

The implemented ReAct flow made 5 model calls and used 907 tokens for S1, adapting after each order, policy, and calculation observation; across the seven scenarios it made 30 calls. It handled missing and cancelled orders using observed tool results rather than assuming a successful refund. A plan-and-execute system could draft a fixed sequence before acting, but this project did not implement or measure that alternative. For this workflow, ReAct's stepwise verification is preferable because refund eligibility and escalation depend on database observations; explicit tool-side checks remain necessary regardless of planning style.

# Limitations and Next Steps

The dataset and policies are synthetic and small; findings should not be presented as real company outcomes or policies. The classifier is a deterministic offline simulator, not a production language model, and strict regexes cover only the specified PII formats. The model benchmark is based on one fixed temporal split and a synthetic six-month order window; it does not establish future production performance. Segment names are heuristic summaries, the Premium retention flag has the post-hoc full-window leakage described above, and correlation/feature importance do not establish causality. The refund ledger and checkpoint files demonstrate safe simulated control flow only; they are not payment integrations or a substitute for production authorization, audit, security, and retention controls. The project is published in the public GitHub repository on the `main` branch. The dataset and policies are synthetic, and the documented results are based on the reproducible pipeline and acceptance tests.

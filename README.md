# App-User Churn Prediction from BigQuery Event Data

![CI](https://github.com/GITHUB_USER/app-engagement-churn/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![BigQuery](https://img.shields.io/badge/Google-BigQuery-4285F4)

Predicts which app users are about to go silent, using only the raw event stream
that a Firebase / Google Analytics 4 app exports to **Google BigQuery**.

This is the set-up most consumer apps actually have: no tidy customer table, just
millions of events (`login`, `payment`, `recharge`, `level_start` ...) keyed by an
anonymous user ID. The project goes from that raw stream to a ranked list of users
to target with a retention campaign, using SQL in BigQuery for the heavy lifting and
scikit-learn for the model.

**Data:** Google's public [Flood-It!](https://developers.google.com/analytics/bigquery/app-gaming-demo-dataset)
demo dataset, obfuscated GA4/Firebase events from a puzzle game on Android and iOS
(June to October 2018), at `firebase-public-project.analytics_153293282.events_*`.

## Key results

RESULTS_PENDING

## Why a game dataset is a fair stand-in for a telecom app

The export schema is identical for every Firebase/GA4 app: one row per event with
`event_name`, `event_params`, `user_pseudo_id`, device and geography. Only the event
names differ. Swapping the list of "journey" events in
[`config.py`](src/appchurn/config.py) points the whole pipeline at a different app.

| Signal in this project | Flood-It! event | Equivalent in a telecom self-care app |
|---|---|---|
| Core usage | `level_start_quickplay` | Login, dashboard / usage view |
| Spending | `spend_virtual_currency`, `in_app_purchase` | Recharge, bill payment, bundle purchase |
| Add-ons | `use_extra_steps` | Data add-on, roaming pack |
| Promotions | `ad_reward` | Offer redemption, rewards |
| Social / referral | `challenge_a_friend` | Refer a friend, family sharing |
| Silence | days since last active | Days since last login or top-up |
| Fading | last 7 days vs. 28-day average | This week's usage vs. normal |
| Leaving for good | `app_remove` | App uninstall, port-out, line deactivation |

The weak spot: a game is not a telecom service, and in-app spend here is tiny. The
method transfers; the specific coefficients do not.

## How it works

```
BigQuery (raw events, one row per event)
   │  SQL: collapse to one row per user per day          ← only step that scans raw data
   │  SQL: weekly snapshot dates T
   │       features from [T-28d, T)   label from [T, T+14d)
   ▼
user-snapshot table (one row per user per snapshot)  ──►  data/user_snapshots.parquet
   │
   │  Python: out-of-time split  (train: Jul–Aug, test: Sep)
   │          grouped cross-validation  (no user in both train and validation folds)
   │          benchmarks: churn rate, and a "recency rule" with no model
   ▼
model ──► gains / lift for campaign sizing, feature importance, stability per week
```

### Churn definition

At each weekly snapshot date **T**:

- **In scope:** users active at least once in the 28 days before T.
- **Features** describe those 28 days: active days, sessions, minutes of engagement,
  counts of each journey event, spend, days since last activity, tenure, the trend of
  the last 7 days against the 28-day average, platform, country.
- **Churned = 1** if the user has *no active event* in the 14 days from T.
  Automatic events (`app_remove`, `os_update`, `app_update` ...) do not count as
  activity: an uninstall is not the user coming back.

This is the standard snapshot design for telecom churn: a fixed observation window,
a fixed future window, and a clean gap between what the model sees and what it
predicts. Day T itself belongs to the future window only, and a test checks this.

### Choices that keep the evaluation honest

- **Out-of-time test set.** The model trains on July and August snapshots and is
  tested on September, as it would be in production. A training snapshot is kept only
  if its 14-day label window ends before the first test snapshot.
- **Grouped cross-validation.** The same user appears in several weekly snapshots,
  so folds are split by user. Otherwise the model would be validated on people it had
  already seen.
- **A real benchmark.** Ranking users by "days since last seen" needs no model and
  is hard to beat. The model is only worth deploying if it clearly does better.

### Cost control in BigQuery

- `_TABLE_SUFFIX` limits the scan to the daily tables in the date range.
- Only the needed columns are read, and events are reduced to one row per user per
  day before anything else.
- Every query is **dry-run first** to report the bytes it will scan, then runs with
  `maximum_bytes_billed` as a hard cap (10 GB by default), so a mistake fails instead
  of costing money.
- Authentication uses Application Default Credentials. No key file exists in the
  repository, and `.gitignore` blocks common credential file names.

## Testing without Google Cloud

The production BigQuery SQL is translated with [sqlglot](https://github.com/tobymao/sqlglot)
and executed in [DuckDB](https://duckdb.org) against small, hand-built GA4-shaped
tables, including the nested `event_params` structure. The tests cover:

- window boundaries, and an event on snapshot day counting as future rather than feature
- uninstalls and other passive events not counting as a return
- journey counts, engagement minutes, revenue, recency and tenure
- an end-to-end run on synthetic users who fade out before leaving
- the cost cap being passed to BigQuery

CI runs all of this on every push without any Google credentials.

## Project structure

```
app-engagement-churn/
├── src/appchurn/
│   ├── config.py          # table, date range, journey events, window design
│   ├── sql/
│   │   ├── user_snapshot_features.sql   # features + labels, BigQuery SQL
│   │   ├── event_profile.sql            # which events exist and how often
│   │   └── __init__.py                  # renders and validates the SQL
│   ├── extract.py         # google-cloud-bigquery: dry run, cost cap, parquet cache
│   ├── features.py        # derived trend features, out-of-time split
│   ├── model.py           # pipelines, recency-rule benchmark, gains/lift
│   ├── train.py           # CV, final evaluation, reports
│   └── plots.py
├── tests/                 # SQL logic in DuckDB, end-to-end pipeline, extract
└── reports/               # metrics, CV, gains, per-week results, figures
```

## How to run

You need a Google Cloud project. The free
[BigQuery sandbox](https://cloud.google.com/bigquery/docs/sandbox) is enough, no
credit card required.

```bash
git clone https://github.com/GITHUB_USER/app-engagement-churn.git
cd app-engagement-churn
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

gcloud auth application-default login                   # one time
export GCP_PROJECT=your-project-id                      # Windows: $env:GCP_PROJECT="your-project-id"

python -m appchurn.extract --dry-run    # how much data each query will scan (free)
python -m appchurn.extract              # run in BigQuery, save data/user_snapshots.parquet
python -m appchurn.train                # train, evaluate, write reports/
pytest                                  # tests, no Google account needed
```

### Using it on another app's export

1. Point `events_table`, `data_start` and `data_end` in `config.py` at the export.
2. Run `python -m appchurn.extract --profile` to list the events and their volumes.
3. Put the business-relevant events in `journey_events`.

Run it inside the data owner's own Google Cloud project: the code moves to the data,
not the other way round.

## License

MIT

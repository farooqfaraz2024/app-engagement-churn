# Architecture

Two views: what this repository runs today, and a reference design for running the
same approach in production at a telecom operator. The reference design is generic and
vendor-neutral: product names are examples of a component type, not a description of
any specific company's systems.

## 1. This repository

```mermaid
flowchart LR
    subgraph GCP["Google Cloud · BigQuery"]
        RAW[("Flood-It! GA4 / Firebase export<br/>5.7M events · 15k users<br/>public dataset")]
        COPY[("Optional copy<br/>your-project.flood_it.events")]
        SQL["Feature + label SQL<br/>events → user-days → weekly snapshots<br/>28-day features · 14-day churn label"]
        BQML["bigquery_ml/ scripts<br/>CREATE MODEL · ML.PREDICT<br/>EXPORT DATA"]
        RAW -.-> COPY
        RAW --> SQL
        COPY -.-> SQL
    end

    subgraph PY["Python · runs on a laptop"]
        EXT["extract.py<br/>google-cloud-bigquery<br/>dry run · 10 GB cost cap"]
        PQ[("user_snapshots.parquet<br/>52,828 rows")]
        TRAIN["train.py · scikit-learn<br/>regular users · out-of-time split<br/>forward-chaining validation<br/>recency-rule benchmark · bootstrap"]
        REP["reports/<br/>metrics · gains · importance<br/>charts"]
        SCORE["score.py<br/>latest snapshot · risk bands<br/>reason codes · at-risk CSV"]
        HTML["docs/index.html<br/>interactive report<br/>GitHub Pages"]
        EXT --> PQ --> TRAIN --> REP
        PQ --> SCORE --> HTML
        REP --> HTML
    end

    subgraph CI["GitHub Actions · no Google credentials"]
        SYN["Synthetic GA4 events"]
        DUCK["Same SQL run in DuckDB<br/>via sqlglot"]
        TESTS["Automated tests<br/>windows · leakage · cost cap<br/>BigQuery ML sync · end-to-end"]
        SYN --> DUCK --> TESTS
    end

    SQL --> EXT
    SQL -. "same template" .-> BQML
    SQL -. "identical SQL" .-> DUCK

    classDef store fill:#eef4fc,stroke:#2a78d6,color:#0b0b0b
    classDef step fill:#fcfcfb,stroke:#52514e,color:#0b0b0b
    class RAW,COPY,PQ,REP,HTML store
    class SQL,BQML,EXT,TRAIN,SCORE,SYN,DUCK,TESTS step
```

## 2. Reference design for a telecom operator (production)

```mermaid
flowchart LR
    subgraph SRC["Sources"]
        APP["Self-care mobile app<br/>Firebase SDK<br/>logged-in user ID set"]
        CRM["CRM / billing<br/>customer ID · plan · consent"]
    end

    subgraph BQ["Data warehouse · e.g. BigQuery"]
        EV[("App events<br/>daily export")]
        IDM[("Identity map<br/>app user → customer ID")]
        FEAT["Daily feature build<br/>scheduled SQL"]
        SCORE["Daily scoring<br/>BigQuery ML or Python job"]
        RISK[("at_risk_customers<br/>score · band · reasons")]
        OUT[("Campaign outcomes<br/>contacted · holdout · churned")]
        EV --> FEAT --> SCORE --> RISK
    end

    subgraph ML["Model lifecycle · monthly"]
        RETRAIN["Retrain on recent weeks"]
        GATE{"Beats current model<br/>and recency rule<br/>out-of-time?"}
        RETRAIN --> GATE
    end

    subgraph ACT["Activation"]
        ELIG["Eligibility<br/>consent · contact caps<br/>10% random holdout"]
        EXP["Daily file or API<br/>e.g. Cloud Storage CSV"]
        NBA["Next-best-action platform<br/>e.g. Pega CDH, Salesforce MC<br/>chooses offer and message"]
        PUSH["Push notification<br/>FCM / APNs"]
        ELIG --> EXP --> NBA --> PUSH
    end

    DASH["Dashboard<br/>Looker Studio / HTML report"]

    APP --> EV
    CRM --> IDM
    CRM --> ELIG
    IDM --> RISK
    RISK --> ELIG
    RISK --> DASH
    PUSH -. "customer responds or not" .-> APP
    NBA --> OUT
    OUT --> RETRAIN
    OUT --> DASH
    GATE -- "yes: promote" --> SCORE
    GATE -. "no: keep current" .-> SCORE

    classDef store fill:#eef4fc,stroke:#2a78d6,color:#0b0b0b
    classDef step fill:#fcfcfb,stroke:#52514e,color:#0b0b0b
    classDef act fill:#fdf0ea,stroke:#eb6834,color:#0b0b0b
    class EV,IDM,RISK,OUT store
    class FEAT,SCORE,RETRAIN,DASH,APP,CRM step
    class ELIG,EXP,NBA,PUSH act
```

### Design decisions

| Decision | Why |
|---|---|
| Score daily on the previous complete day | The daily app export lands the following day; scoring partial data understates activity and inflates risk. |
| Identity map before activation | `user_pseudo_id` identifies an app installation, not a customer. The activation platform needs a customer or account ID. |
| Eligibility outside the model | Consent, opt-outs and contact-frequency caps are rules owned by marketing and compliance, not something a model should learn. |
| Random holdout of at-risk customers | Without a no-contact control group, a campaign's effect on churn cannot be measured. Those outcomes also enable a future uplift model. |
| Promotion gate on retraining | A new model replaces the current one only if it wins on out-of-time data, including against the simple recency rule. |
| Scores, not raw events, leave the warehouse | Personal event data stays in the warehouse; only the at-risk list with the fields the campaign needs is exported. |

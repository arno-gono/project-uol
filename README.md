# project-uol
CSM500 - University of London Project: agentic root-cause diagnosis for data-quality incidents.

## What it does

A pipeline that tests whether an AI agent can find data-quality errors in a relational database:

1. **Import**: a Kaggle dataset is downloaded and loaded into SQLite. The rows are split in two: 97% considered clean, 3% set aside as a new batch.
2. **Calibration**: the clean data is profiled into a JSON file (datatypes, nulls, distributions, keys, correlations, plus a machine learning layer: PCA, KMeans, disguised missing values).
3. **Error injection**: errors drawn at random from ten types are injected into the new batch (orphan foreign keys, duplicates, distribution shifts, correlation breaks...).
4. **Agent investigation**: a Claude model investigates the new batch with two tools, one running SQL queries and one reading the calibration, and reports its findings.
5. **Reconciliation**: the findings are matched against the errors injected and scored.

Everything is logged as JSON in `app/logs/` (created on the first run).

## Setup

Python 3.14 was used.

```commandline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the root of the project holding an Anthropic API key:

```commandline
ANTHROPIC_API_KEY=your-key-here
```

The dataset is set in `app/config.py` (`KAGGLE_DATASET_NAME`, `airbnb/seattle` by default). Datasets are downloaded with kagglehub.

## Run it from the dashboard

From the root of the project:

```commandline
streamlit run app.py
```

The **Investigation** tab runs the pipeline: tick the stages to run (Import Data, Run Calibration, Inject Errors, Call Agent, Reconcile Logs), pick the model in the sidebar and click Run Investigation. On a fresh clone, tick all five for the first run. To compare models on the same errors, run again with only Call Agent and Reconcile Logs ticked.

The **Results** tab shows the score, the cost and the error types found or missed over all investigations.

## Run it from the command line

Set the stages and the model at the bottom of `app/run.py` (the `if __name__ == "__main__":` block), then from the root of the project:

```commandline
python -m app.run
```

## Settings

In `app/config.py`:

- `AGENT_MODEL`: default model (the models available are the ones priced in `AGENT_MODELS_COSTS`)
- `AGENT_READS_ML_CALIBRATION`: whether the agent receives the machine learning layer of the calibration
- `AGENT_MAX_TOKENS`: maximum tokens per response
- `AGENT_MAX_ROWS_RETURNED`: maximum rows returned by one SQL query

## Tests

```commandline
python -m pytest test
```

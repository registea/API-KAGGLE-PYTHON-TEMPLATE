# API-KAGGLE-PYTHON-TEMPLATE

Develop Python locally and run selected pipeline nodes on Kaggle.
The current IEEE-CIS example joins competition data into a flat CSV, then fits
a one-feature classifier. An optional score node generates test-set predictions. This is a pipeline template, not a competition-ready model.

## Local setup

Activate your Conda environment (Python 3.11/3.12 for the pinned dependencies),
then run from the project root:

```powershell
python -m pip install -r src/training/environment/requirements.txt
python -m pip install -e ".[submit,dev]"
Copy-Item .env.example .env
python -m pytest
```

GitHub Actions runs formatting, linting, type-checking and tests on Python 3.11
and 3.12 for pull requests and pushes to the main development branches. The
workflow is defined in `.github/workflows/ci.yaml`.

Alternatively, create and activate a virtual environment first.
The editable install registers training-submit and training-run.
Update the project name/description in pyproject.toml when reusing this template.

## Kaggle access

Copy `.env.example` to `.env`, then replace its example values with your
Kaggle credentials:

```dotenv
KAGGLE_API_TOKEN=your-token-from-kaggle
KAGGLE_KERNEL_ID=your-username/training-template
```

Generate a token in [Kaggle API settings](https://www.kaggle.com/settings/api).
Accept the rules on the [IEEE-CIS competition page](https://www.kaggle.com/competitions/ieee-fraud-detection)
using the account associated with your token. The metadata already attaches
ieee-fraud-detection and defaults to private, CPU-only, offline execution.

Kernel identity resolves in this order:

1. --kernel-id owner/slug
2. KAGGLE_KERNEL_ID
3. KAGGLE_USERNAME plus KAGGLE_KERNEL_SLUG (both required)
4. The source metadata's id

Environment variables override local .env values. On build agents with BUILD_REASON
set or CI=true, 1 or yes, dotenv is skipped. Inject credentials as secret variables.
The token is required for remote commands and is never embedded in generated code
or command arguments. Empty/example tokens are rejected; Kaggle validates credentials
when connecting. Local preparation does not require credentials.

## Run both pipeline nodes

```powershell
python -m training.main submit --nodes data_process fit
```

Both stages run sequentially inside **one Kaggle job**, in dependency order:

| Node | Action | Output |
| --- | --- | --- |
| data_process | Left-join train_transaction.csv and train_identity.csv on TransactionID | training.csv |
| fit | Read the prepared CSV and fit a one-feature classifier | model.joblib and metrics.json |

The processing node retains transactions without identity records and rejects
duplicate/null join keys. By default it reads 10,000 transaction rows plus the
identity table. Use --max-rows 0 to read all transaction rows.

The fitting node selects the first numeric column excluding TransactionID and
isFraud (typically TransactionDT in the competition's standard column order).
It uses a stratified random holdout and fits median imputation, scaling and
class-balanced logistic regression on the training split. Metrics include ROC AUC
and average precision. The saved model is refitted on all selected labelled rows after evaluating that training split.
Choose meaningful features and validation for a real competition solution.

Outputs are saved under /kaggle/working on Kaggle or outputs/ locally.
--output-dir overrides the location. Reusing a directory overwrites matching
output filenames, so use distinct directories for separate local experiments.

## Run individual nodes

Process data only:

```powershell
python -m training.main submit --nodes data_process
```

Fit from an earlier kernel's prepared output:

```powershell
python -m training.main submit --nodes fit --processed-data /kaggle/input/data-preparation/training.csv --kernel-sources your-username/data-preparation
```

The upstream kernel must already have completed successfully and saved training.csv.
Use the actual attached file path under /kaggle/input. For separate processing and
fitting jobs, use distinct kernel IDs and matching titles:

```powershell
python -m training.main submit --kernel-id your-username/data-preparation --title "Data Preparation" --nodes data_process
python -m training.main status --kernel-id your-username/data-preparation

# After successful processing:
python -m training.main submit --kernel-id your-username/model-fitting --title "Model Fitting" --nodes fit --processed-data /kaggle/input/data-preparation/training.csv --kernel-sources your-username/data-preparation
```

The launcher does not automatically schedule or wait for upstream jobs.
It embeds --nodes, --input-dir, --output-dir, --processed-data, --model-path and --max-rows into
the generated remote entry point. Remote paths must exist on Kaggle; local data
is not uploaded automatically.

Omitting --nodes uses job/cfg/training.yaml (both nodes by default).
A fit-only run without --processed-data looks for training.csv in the output
directory. Missing inputs fail explicitly. Combining --processed-data with
data_process is rejected to avoid ambiguity.

## Develop locally

Download and unzip competition CSV files, then run:

```powershell
python -m training.job.execute --nodes data_process fit --input-dir data/raw/ieee-fraud-detection --output-dir outputs/example --max-rows 1000
python -m training.job.execute --nodes fit --processed-data outputs/example/training.csv --output-dir outputs/refit
```

execute.py routes the pipeline; data_process.py and fit.py expose run functions
that return output paths. Shared helpers belong in job/utils/.
Use data/raw/ for local source data, data/processed/ for derived data, and
exploration/ for notebooks. Dataset contents are Git-ignored.

Tests under src/training/tests/ use synthetic data to verify joins, fitting,
routing, packaging, authentication, overrides and console filtering.

## Submission and logs

| Command/option | Behaviour |
| --- | --- |
| training-submit prepare | Generate code and metadata locally. |
| training-submit submit | Upload, start execution and follow logs. |
| training-submit status | Check the latest run for the selected kernel. |
| training-submit output | Download saved outputs to the configured local output directory. |
| --no-follow | Return immediately after submission. |
| --raw-logs | Display the complete Kaggle log stream. |

The default filter hides recognised notebook-rendering messages, preserving job
errors and unknown output. Ctrl+C stops the local viewer, not the Kaggle job.
Log streaming ending does not prove training succeeded: check status.
If streaming fails after submission, check the existing run before resubmitting.

## Configuration and overrides

| File | Purpose |
| --- | --- |
| src/training/cfg/config.yaml | Local build/output paths and metadata location |
| src/training/environment/kernel-metadata.json | Kernel identity, compute and data attachments |
| src/training/job/cfg/training.yaml | Node defaults, input filenames, target and model settings |
| src/training/environment/requirements.txt | Pinned data science dependencies |
| pyproject.toml | Local package metadata and tools |

Submission paths resolve against the project root; use --project-root when needed.
--config-path selects submission YAML on training.main, or training YAML on
training.job.execute. Remote jobs use the packaged training YAML plus forwarded
pipeline arguments.

Override metadata without editing files:

| Argument | Setting |
| --- | --- |
| --metadata-path path/to/file.json | Alternative metadata file |
| --kernel-id owner/slug and --title "Title" | Identity/title |
| --private / --no-private | Visibility |
| --enable-gpu / --no-enable-gpu | GPU |
| --enable-internet / --no-enable-internet | Internet access |
| --machine-shape identifier | Supported accelerator |
| --dataset-sources owner/data ... | Dataset attachments |
| --competition-sources competition ... | Competition attachments |
| --kernel-sources owner/kernel ... | Outputs from earlier kernels |
| --model-sources model-reference ... | Model attachments |

List flags replace configured lists; passing no values clears the list.
Overrides apply to the current invocation only. Repeat identity overrides for
status/output or persist them in .env. Source metadata stays unchanged.

The packager embeds job/ Python, YAML and JSON files in build/kaggle/run.py.
It preserves imports and always creates a Python script. Local submission helpers
and credentials are excluded. Edit src/ and prepare again; generated files are overwritten.

Kaggle submission does not automatically install requirements.txt. Ensure the
libraries your nodes import are available remotely. Extra installs need explicit
setup, with internet enabled when permitted or dependency wheels attached for offline use.
Direct dependencies are pinned for Python 3.11/3.12; transitive dependencies are not locked.

Local tools include pytest, Ruff and pre-commit. Git hooks are not configured.
Run the complete local checks with:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest
```

For Windows corporate certificate issues, the local fix previously used was
python -m pip install pip-system-certs==5.3; see
[pip-system-certs documentation](https://pypi.org/project/pip-system-certs/).

## Start a new Kaggle project

When creating a project from this template:

1. Update the project name and description in `pyproject.toml`.
2. Set the kernel identity, title, compute options and attached sources in
   `src/training/environment/kernel-metadata.json`.
3. Replace the IEEE-CIS settings in `src/training/job/cfg/training.yaml`.
4. Replace the example logic in `data_process.py`, `fit.py` and `score.py`.
5. Update the synthetic tests to describe the new data and expected artefacts.
6. Copy `.env.example` to `.env` and add your own Kaggle credentials.
7. Run locally on a small sample, then use `training-submit prepare` to inspect
   the generated Kaggle package before submitting it.

See [the component guide](docs/docs.md) for the execution flow, configuration
precedence and the responsibilities of each module.

## Project structure

```text
README.md
.env.example
pyproject.toml
docs/docs.md
src/training/
  main.py                         # Submission and console logs
  cfg/config.yaml
  environment/
    kernel-metadata.json
    requirements.txt
  utils/                          # Settings, packaging, metadata and log filtering
  job/
    execute.py                    # Node routing
    data_process.py               # Join and save data
    fit.py                        # Fit, evaluate and refit model
    score.py                      # Predict full test set and write submission.csv
    cfg/training.yaml
    utils/                        # Shared arguments, logging and configuration
  tests/
data/
  raw/
  processed/
exploration/
```

Generated build/ and outputs/ are Git-ignored. General notes belong in docs/.
Python files use named comment blocks, setup_logger() and get_config().
YAML and JSON load into plain dictionaries.

## Score the test set and optionally submit predictions

Launching a compute job and submitting predictions to a competition are separate
actions. The leaderboard step is opt-in.

### Train, score and submit in one command

```powershell
python -m training.main submit --submit-predictions
```

With no explicit --nodes, this selects data_process, fit and score.
The launcher waits for a successful Kaggle run, downloads submission.csv and
submission-manifest.json into a new local outputs/predictions-* directory, verifies
the run ID, competition, checksum and row count, then calls the Kaggle competition
submission command once. Credentials remain on the local machine.

Keep the local console running until the workflow completes. --no-follow disables
log viewing but still waits when --submit-predictions is enabled.
--submission-timeout sets the status polling limit in seconds (default 43200).
An interrupted workflow does not cancel the compute job. Check Kaggle's submission
history before retrying if a network error occurs during leaderboard upload.

Optional settings:

```powershell
python -m training.main submit --submit-predictions --competition ieee-fraud-detection --submission-message "One-feature baseline"
```

The competition defaults to the single attached competition and must match
competition in job/cfg/training.yaml. If --nodes is explicitly provided, it must
include score. Outputs for this workflow must be under /kaggle/working so Kaggle
retains them. A mismatched run artefact causes an error rather than submitting
an older prediction file; avoid concurrent runs using the same kernel ID.

Competition eligibility, deadlines and submission limits still apply. Generating
a CSV remains possible even if Kaggle no longer accepts entries for a competition.

### Generate predictions without a leaderboard submission

```powershell
python -m training.main submit --nodes data_process fit score
```

The score node reads the entire test set, regardless of --max-rows (which limits
training transactions only). It uses the saved pipeline's feature names and
preprocessing, joining test identity data if needed, and writes these files:

- submission.csv: TransactionID and isFraud probability, in sample_submission.csv order.
- submission-manifest.json: run identity, competition, file checksum and dimensions.

Scoring validates unique IDs, complete test coverage and finite probabilities
between zero and one. It fails if test columns or sample submission IDs do not match.

### Score using an existing model

```powershell
python -m training.main submit --nodes score --model-path /kaggle/input/model-fitting/model.joblib --kernel-sources your-username/model-fitting
```

Add --submit-predictions to also enter those predictions into the competition.
--model-path cannot be combined with the fit node, which creates a fresh model.
For local scoring, pass the same node and data/model paths to
python -m training.job.execute.

Default training still uses at most 10,000 transaction rows. Use --max-rows 0 for
all training transactions. Holdout metrics are measured before the pipeline is
refitted on every selected labelled row; model.joblib stores that refitted pipeline.

Scoring parameters live in job/cfg/training.yaml: competition,
test_transaction_file, test_identity_file and sample_submission_file.
Ordinary submit without --submit-predictions never sends anything to the leaderboard.

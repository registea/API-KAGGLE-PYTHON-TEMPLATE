# Component guide

This guide explains how the template moves code and configuration from a local
checkout into a Kaggle script. The [README](../README.md) contains setup and
command examples.

## Execution flow

There are two entry points:

- `training-submit` calls `training.main`. It prepares a Kaggle package and can
  push it, inspect its status, download outputs or submit verified predictions.
- `training-run` calls `training.job.execute`. It runs the pipeline directly,
  either locally or inside the generated Kaggle script.

A remote run follows this sequence:

1. `training.main` reads local settings, submission configuration and kernel
   metadata.
2. Command-line overrides are applied and validated.
3. `training.utils.package.prepare_kernel` collects the remote job modules and
   job configuration into a generated `run.py`.
4. The Kaggle CLI uploads `run.py` and `kernel-metadata.json`.
5. On Kaggle, `run.py` recreates the embedded `training.job` package and calls
   `training.job.execute.main`.
6. The selected nodes run in dependency order and write retained artefacts below
   `/kaggle/working`.

Modules under `training/job/` are sent to Kaggle. Modules under
`training/utils/` and the local `training/main.py` remain on the local machine,
along with credentials and submission controls.

## Local launcher

### `training/main.py`

This is the controller for Kaggle operations. It resolves the project root,
loads configuration, validates authentication, prepares the kernel package and
invokes the Kaggle CLI. It also coordinates log streaming and optional
leaderboard submission.

The `prepare` action performs packaging without contacting Kaggle. Inspect
`build/kaggle/run.py` and `build/kaggle/kernel-metadata.json` when diagnosing
packaging or metadata problems.

### Local utility modules

- `settings.py` loads credentials from `.env` without overriding existing
  environment variables. It skips dotenv files in CI and build-agent
  environments. Credentials are never written into the generated package.
- `metadata.py` combines checked-in kernel metadata with command-line and
  environment overrides. It validates identity, compute settings and attached
  sources without modifying the source JSON.
- `package.py` collects Python, YAML and JSON below `training/job/` in a stable
  order and embeds them in the bootstrap script. Generated files should not be
  edited directly.
- `logs.py` streams Kaggle output and filters recognised notebook-conversion
  noise. Use `--raw-logs` to display everything.
- `competition.py` waits for a successful job, downloads into a fresh
  directory and verifies the prediction manifest before allowing a competition
  submission.

## Remote pipeline

### `training/job/execute.py`

The remote entry point selects nodes and resolves their paths. Nodes always run
in the order `data_process`, `fit`, then `score`, regardless of their order on
the command line. A node may consume an artefact from an earlier job when the
corresponding path override is supplied.

Node imports occur only when needed. This keeps routing simple and makes missing
dependencies surface at the node that requires them.

### Pipeline nodes

- `data_process.py` reads the example files, validates join keys, performs a
  left join and writes `training.csv`. In a new project, keep this module
  responsible for deterministic data preparation.
- `fit.py` evaluates the example scikit-learn pipeline, then refits it on all
  selected labelled rows. It writes `model.joblib` and `metrics.json`. Replace
  its features and validation strategy for each competition, while keeping
  preprocessing inside the serialised model.
- `score.py` loads the model, prepares test rows and aligns probabilities with
  the sample submission. It validates identifiers, coverage and probability
  values before writing `submission.csv` and its manifest.

The shared `job/utils/arguments.py` defines options accepted locally and
remotely. `utils.py` loads YAML and JSON configuration, while `logging.py`
provides a common logger without removing handlers installed by callers.

## Configuration and precedence

| File | Responsibility |
| --- | --- |
| `src/training/cfg/config.yaml` | Local build, download and metadata paths |
| `src/training/environment/kernel-metadata.json` | Kaggle identity, compute settings and attached sources |
| `src/training/job/cfg/training.yaml` | Pipeline nodes, data paths, filenames and model settings |

Command-line values override matching checked-in settings for the current run.
Kernel identity can also come from environment variables. Pipeline arguments
supplied to `training-submit` are embedded in `run.py`, so the remote process
receives the same node selection and path overrides.

Local paths are interpreted from the project checkout. Kaggle inputs normally
sit below `/kaggle/input`, and retained remote outputs must be written below
`/kaggle/working`. Local datasets are not uploaded automatically.

## Artefacts and partial runs

| Producer | Artefact | Typical consumer |
| --- | --- | --- |
| `data_process` | `training.csv` | `fit` |
| `fit` | `model.joblib` | `score` |
| `fit` | `metrics.json` | Local review or downloaded outputs |
| `score` | `submission.csv` | Kaggle competition submission |
| `score` | `submission-manifest.json` | Local submission verification |

For a partial run, attach or download the upstream artefact and pass its Kaggle
path using `--processed-data` or `--model-path`. The launcher does not schedule
dependencies across separate kernels.

## Adapting the template

Start with the three node modules and `training.yaml`. Keep node interfaces
small: accept paths and configuration, write explicit artefacts, and return the
main output path. Add shared remote code below `training/job/utils/`; add local
Kaggle-control code below `training/utils/`.

Update the synthetic tests alongside each node. They use small generated
datasets, so they check pipeline behaviour without requiring competition
downloads or Kaggle credentials.

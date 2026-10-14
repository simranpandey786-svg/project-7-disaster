# Disaster Tweet Classification

A machine-learning project that classifies a tweet as a real disaster-related tweet or not. It includes data exploration, TF-IDF models, optional DistilBERT fine-tuning, evaluation reports, and a Flask web app.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11 recommended
- Internet access for installing packages and (if used) downloading the DistilBERT base model
- The labelled tweet dataset described below

## 1. Download and place the dataset

Download the **NLP with Disaster Tweets** dataset from Kaggle. Put its labelled `train.csv` in the project's `data` folder:

```text
data/train.csv
```

The CSV must contain `text` and `target` columns. The Kaggle file also includes columns such as `id`, `keyword`, and `location`; these are okay. `target` must use `1` for a real disaster and `0` for not a real disaster.

The dataset is not included in this repository. You may need to sign in to Kaggle and accept its dataset terms before downloading it.

## 2. Create an environment and install dependencies

Open PowerShell in the project folder (the folder containing `requirements.txt`) and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use Command Prompt instead:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency install includes PyTorch and Transformers for the optional DistilBERT model and may take a while. If `py -3.11` is not available, install Python 3.11 and try again.

## 3. Prepare data and train the models

Run these commands from the project folder, with the virtual environment activated:

```powershell
python explore.py
python prepare_data.py
python train_tfidf.py
```

This creates plots in `reports/`, train/test CSV splits in `data/`, and two trained models in `models/`. The TF-IDF training uses 5-fold cross-validation and can take several minutes.

### Optional: train DistilBERT

DistilBERT fine-tuning needs more memory and time; a CUDA-capable GPU is recommended. The script downloads the base model on its first run.

```powershell
python train_distilbert.py
```

You can adjust training with options such as `--epochs`, `--batch-size`, and `--max-length`. Run `python train_distilbert.py --help` to see all options.

## 4. Evaluate

After at least one model has been trained:

```powershell
python evaluate.py
```

Metrics, confusion matrices, ROC/precision-recall plots, and misclassified examples are written to `reports/`.

## 5. Run the web app

```powershell
python app.py
```

Open <http://127.0.0.1:5000> in a browser. The app lists whichever trained models it finds in `models/`. To check that it is running, open <http://127.0.0.1:5000/healthz>.

The prediction endpoint accepts JSON, for example:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/predict -Method Post -ContentType "application/json" -Body '{"text":"Forest fire near the town"}'
```

Stop the local server with `Ctrl+C`.

## Important when sharing or submitting

The source code alone is not enough to make predictions. The `data/train.csv` dataset and at least one trained model must be present, or the recipient must download the dataset and run the preparation/training steps above. Without a model, the app can start but prediction returns HTTP 503 with a message that no trained model was found.

Do not include `.venv/` in a ZIP or Git push; recipients should create their own environment using the steps above. The DistilBERT model is large and optional; the TF-IDF models are sufficient to run the app.

## Troubleshooting

- **`data/train.csv not found`**: download the labelled dataset and check the path and filename.
- **CSV missing `text` or `target`**: use the labelled Kaggle `train.csv`, not `test.csv`.
- **`Run prepare_data.py first`**: run `python prepare_data.py` before training or evaluation.
- **`No trained models found` / HTTP 503**: run `python train_tfidf.py` (or train DistilBERT) successfully, then restart the app.
- **PowerShell activation is blocked**: follow the Command Prompt setup commands above.
- **DistilBERT download/training fails or is too slow**: skip the optional DistilBERT step and use the TF-IDF models.

## Project structure

```text
app.py                 Flask web app
common.py              Shared paths, cleaning, and feature helpers
explore.py             Dataset summaries and plots
prepare_data.py        Deduplication and stratified train/test split
train_tfidf.py         TF-IDF + Logistic Regression / Random Forest training
train_distilbert.py    Optional DistilBERT fine-tuning
evaluate.py            Held-out test evaluation and reports
predictors.py          Loads trained models for evaluation and the web app
data/                  Input CSV and generated data splits
models/                Trained model files
reports/               Generated plots, metrics, and training reports
templates/index.html   Web interface
```

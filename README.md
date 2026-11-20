# Patient Health Record Dashboard

A Streamlit dashboard for exploring patient records, summarizing vital signs, reviewing health alerts, and generating academic Random Forest predictions.

## Features

- Patient, disease, vital-sign, and alert views
- Interactive filtering and dataset quality summaries
- Exact patient-number search and paginated browsing for large datasets
- Data accuracy summary alongside duplicate and missing-value checks
- Random Forest predictions with class probabilities and feature importance
- A visible disclaimer that model output is not a medical diagnosis

## Run locally

Use Python 3.11 or newer. From the project directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app reads the spreadsheet from `data/` and the model artifacts from `model/`, regardless of the current working directory.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub. The dataset and small feature list are included; the 699 MB `random_forest_model.pkl` and local Streamlit secrets are excluded by `.gitignore`.
2. Upload `model/random_forest_model.pkl` to a trusted HTTPS object-storage location. Keep the URL stable and accessible to the app.
3. Calculate the artifact checksum locally:

	```powershell
	Get-FileHash .\model\random_forest_model.pkl -Algorithm SHA256
	```

4. Create the Streamlit Community Cloud app from the repository with `app.py` as the entry point.
5. In the app's **Settings > Secrets**, add the model URL and checksum:

	```toml
	MODEL_URL = "https://your-trusted-storage.example/model.pkl"
	MODEL_SHA256 = "paste-the-64-character-sha256-value-here"
	```

The model downloads only when a prediction is requested, then is cached in the app's temporary storage. Downloads require HTTPS, are limited to 1 GiB, and are rejected if the SHA-256 checksum differs. The hosting environment still needs enough memory to load the model. The bundled model is pickle-based, so only host and load artifacts from a source you trust.

## Tests

Run the unit tests and section-level Streamlit smoke tests with:

```powershell
python -m unittest discover -v
```

## Model and data safety

The `.pkl` model files are Python pickle-based artifacts. Only use model files from a trusted source; loading an untrusted pickle can execute code. Keep patient data private and do not treat model predictions as clinical advice. Install a scikit-learn version compatible with the one used to train the bundled model if model loading reports a version warning or error.
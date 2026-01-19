# MEM Shirt Size Predictor — End-to-End Walkthrough (A → B)

This document explains how the project goes from **raw user inputs** (height, weight, age, etc.) to a **shirt size recommendation** (S/M/L/XL/XXL), including how the training pipeline works and how the API serves predictions.

---

## 1) What the system does (high level)

**Goal:** Given a user profile:

- `height_cm`
- `weight_kg`
- `age`
- `gender` (female/male/other)
- `fit_preference` (slim/regular/oversized)
- `build` (lean/average/athletic/curvy)

…the system predicts a recommended T‑shirt size in:

- `S`, `M`, `L`, `XL`, `XXL`

**Two phases:**

1. **Offline training** creates a saved model artifact: `app/model.joblib`
2. **Online inference** loads that artifact and serves predictions via FastAPI

---

## 2) Repository map (what each file is responsible for)

- `train_model.py`
  - Generates training data (synthetic), trains a scikit‑learn pipeline, evaluates it, and saves `app/model.joblib`.
- `app/feature_utils.py`
  - Defines the _feature order_ and the function used inside the sklearn pipeline to turn dict rows into a matrix.
- `app/server.py`
  - FastAPI service that loads the trained model and exposes HTTP endpoints (`/health`, `/model-info`, `/predict`).
- `client.py`
  - Example script that sends requests to the API and prints the JSON responses.
- `Dockerfile`
  - Containerizes the API server and runs it using Uvicorn.
- `requirements.txt`
  - Python dependencies.

---

## 3) Training pipeline (A → B)

### A. Input to training

Training starts in `train_model.py`, inside `main()`:

1. It generates a dataset using `generate_synthetic_dataset(...)`.
2. It trains a model using `train_model(...)`.
3. It builds a `ModelArtifact` (metadata + trained pipeline + metrics).
4. It saves the artifact to `app/model.joblib` with `joblib.dump(...)`.

### B. Synthetic dataset: `generate_synthetic_dataset(n, seed)`

**Purpose:** create realistic‑looking training rows and corresponding size labels.

**Output:**

- `rows`: a list of dictionaries (one dict per “user”)
- `y`: a NumPy array of size labels (`S`/`M`/`L`/`XL`/`XXL`)

**What it generates:**

- Discrete fields sampled from distributions:
  - `gender`: `female`, `male`, `other`
  - `fit_preference`: `slim`, `regular`, `oversized`
  - `build`: `lean`, `average`, `athletic`, `curvy`
- Continuous fields generated with randomness + constraints:
  - `age`: integer range
  - `height_cm`: normal distribution adjusted by `gender`, clipped to realistic range
  - `weight_kg`: computed via an internal BMI-like mechanism, adjusted by `build`, clipped

**Label creation logic:**

- A “chest circumference proxy” (`chest_cm`) is computed from height/weight plus adjustments for gender/build/fit.
- `chest_cm` is then mapped to a size using `_size_from_chest_cm(...)`.

### C. Helper functions used during dataset creation

- `_clip(a, lo, hi)`
  - Clamps a numeric array to a min/max range.
  - Used to keep generated values (height, BMI, weight, etc.) within realistic bounds.

- `_size_from_chest_cm(chest_cm)`
  - Converts a numeric chest measurement into one of the size labels.
  - Internally uses `np.digitize` with fixed thresholds (“bins”).

**Important note (present well):** The size thresholds are intentionally simple; in a real merch sizing pipeline, these should be replaced with real supplier measurements and possibly per-fit/per-gender sizing rules.

### D. Model training: `train_model(rows, y, seed)`

**Purpose:** fit a scikit‑learn pipeline that accepts row dicts and outputs a predicted size.

**Key steps:**

1. **Train/test split**
   - Uses `train_test_split` with `stratify=y` so each size class stays proportionally represented.

2. **Convert dicts → matrix**
   - A `FunctionTransformer(rows_to_matrix, validate=False)` is the first stage in the pipeline.
   - That means at inference time we can pass `[{...row...}]` and the pipeline can still handle it.

3. **Preprocess numeric vs categorical columns**

   The feature vector order is fixed by `FEATURE_ORDER`:
   1. `height_cm` (numeric)
   2. `weight_kg` (numeric)
   3. `age` (numeric)
   4. `gender` (categorical)
   5. `fit_preference` (categorical)
   6. `build` (categorical)

   Preprocessing uses a `ColumnTransformer`:
   - Numeric indices `[0, 1, 2]` → `StandardScaler()`
   - Categorical indices `[3, 4, 5]` → `OneHotEncoder(handle_unknown="ignore")`

4. **Classifier**
   - `LogisticRegression(multi_class="multinomial", max_iter=2000)`
   - Trained to predict one of the size labels.

5. **Evaluation**
   - Computes `accuracy_score` and `classification_report` on the holdout test set.
   - Returns both the trained pipeline and a `metrics` dict.

### E. Why `app/feature_utils.py` exists

`rows_to_matrix(...)` is intentionally placed in `app/feature_utils.py` instead of being nested in the training script.

Reason: scikit‑learn pipelines are pickled; when the API server loads the artifact later, it needs to import the same function by module path. Keeping it in an importable module ensures the pipeline is reliably unpicklable in the FastAPI process.

---

## 4) Saved model artifact (what gets persisted)

The model is saved to:

- `app/model.joblib`

What’s inside is a dictionary created from the `ModelArtifact` dataclass:

- `version`: e.g. `shirt-size-v1`
- `trained_at`: timestamp
- `labels`: list of possible outputs
- `feature_schema`: JSON-schema-like contract describing expected input fields
- `pipeline`: the full scikit‑learn pipeline
- `metrics`: evaluation results (accuracy + per-class report)

This artifact is what the API loads at runtime.

---

## 5) API serving (how inference works)

The API service is implemented in `app/server.py` using FastAPI.

### A. Model loading at startup

At import time:

- `_load_artifact()` checks for `app/model.joblib` and loads it via `joblib.load(...)`.
- `_ARTIFACT` and `_PIPELINE` are stored as module-level globals.

This means:

- If the artifact is missing, the server fails fast (clear error).
- Predictions are fast because the model is already loaded.

### B. Request validation with Pydantic

`PredictRequest` defines the expected fields and their valid ranges.

Examples:

- `height_cm`: float, 120–230
- `weight_kg`: float, 30–250
- `age`: int, 10–100
- Enums for `gender`, `fit_preference`, `build`

This keeps the API stable and prevents invalid payloads from reaching the model.

### C. Endpoints (what each one does)

#### `GET /`

Returns a simple message and a pointer to docs.

#### `GET /health`

Returns:

- service status
- whether model is loaded
- model version
- training timestamp

Use this for monitoring.

#### `GET /model-info`

Returns:

- model metadata
- feature schema
- stored evaluation metrics

This is useful in a presentation to prove what the model expects and what performance it got on the holdout set.

#### `POST /predict`

**Main inference endpoint.** Supports two payload formats:

1. **Preferred (explicit fields):**

```json
{
  "height_cm": 175,
  "weight_kg": 75,
  "age": 24,
  "gender": "male",
  "fit_preference": "regular",
  "build": "average"
}
```

2. **Backward-compatible (legacy) format:**

```json
{
  "features": [175, 75, 24, "male", "regular", "average"]
}
```

**Inference steps inside `predict(...)`:**

1. Validate/normalize request (either format becomes a `PredictRequest`).
2. Convert the request to a plain dict (`row`).
3. Call:
   - `_PIPELINE.predict_proba([row])` to get class probabilities.
4. Build a `probabilities` dict mapping label → probability.
5. Choose the largest probability as:
   - `recommended_size`
   - `confidence`

**Response shape (`PredictResponse`):**

```json
{
  "recommended_size": "M",
  "confidence": 0.62,
  "probabilities": { "S": 0.1, "M": 0.62, "L": 0.22, "XL": 0.05, "XXL": 0.01 },
  "model_version": "shirt-size-v1"
}
```

---

## 6) Example client flow

`client.py` shows how to call the API:

1. Set the URL (default is `http://0.0.0.0:8000/predict`).
2. Define example user payloads.
3. Send `POST` requests using `requests.post(..., json=payload)`.
4. Print the returned JSON.

This script is ideal for a short live demo.

---

## 7) Running the system

### Option A: Local (recommended for quick iteration)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model (creates `app/model.joblib`):

```bash
python train_model.py
```

3. Start the API:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

4. Open docs:

- `http://0.0.0.0:8000/docs`

5. Or run the client:

```bash
python client.py
```

### Option B: Docker

1. Build the image:

```bash
docker build -t mem-shirt-size .
```

2. Run the container:

```bash
docker run --rm -p 8000:8000 mem-shirt-size
```

Then use:

- `http://0.0.0.0:8000/docs`

---

## 8) Presentation tips (how to explain it cleanly)

If you only have a few minutes, a solid “tight” storyline is:

1. Inputs (what the user provides)
2. Feature order and conversion (`rows_to_matrix`)
3. Preprocessing (scaling + one-hot encoding)
4. Classifier (multinomial logistic regression)
5. Artifact saving (`model.joblib` contains pipeline + metadata)
6. API endpoints (health/model-info/predict)
7. Demo request → response with probabilities and confidence

---

## 9) Notes / limitations (say this confidently)

- The dataset is synthetic, so reported accuracy is only meaningful for this synthetic distribution.
- Size thresholds are fixed; real production sizing should use supplier charts and real user feedback.
- The API validates inputs, but the model can still be biased by how the synthetic generator was designed.

---

## 10) Quick demo payloads

Use these in `/docs` → `POST /predict`:

```json
{
  "height_cm": 165,
  "weight_kg": 58,
  "age": 22,
  "gender": "female",
  "fit_preference": "regular",
  "build": "lean"
}
```

```json
{
  "height_cm": 178,
  "weight_kg": 82,
  "age": 28,
  "gender": "male",
  "fit_preference": "regular",
  "build": "athletic"
}
```

```json
{
  "height_cm": 172,
  "weight_kg": 95,
  "age": 35,
  "gender": "male",
  "fit_preference": "oversized",
  "build": "average"
}
```

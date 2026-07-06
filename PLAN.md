# Project Plan: Heart Disease Risk Model

**Status:** Delivered (model v1.2). Living document, updated as the system evolves.

This plan is written as a story, because that is how the project actually
unfolded: it started from an existing analysis, I read that analysis critically,
and each thing I built was a direct answer to a specific weakness I found in it.

---

## 1. Where it started: an existing analysis

The project did not start from a blank page. It started from a completed
classification study of the same heart-disease dataset (kept in
`reference/baseline_classifier.ipynb`). That notebook is a solid piece of
analysis: it de-duplicates the data, does thorough EDA and feature selection,
and compares five models (Logistic Regression, Random Forest, SVM, KNN, Decision
Tree), tuning the strongest with grid search. Random Forest came out on top at
roughly 85% accuracy, F1 0.87, AUC 0.90, and the notebook ends by predicting on
one hand-written "simulated patient" to show the idea working.

As a piece of data science, it does its job. My question was different: **what
would it take to turn this analysis into something that could actually run as a
system, and what is it quietly getting wrong?** Reading it with that lens, I
found a set of gaps. The rest of this plan is what I decided to do about each.

---

## 2. What I found reading the baseline, and what I decided to do

I grouped the weaknesses into two kinds: **things that were missing** (it was an
analysis, not a system) and **things that were wrong** (a modelling flaw). Each
one below is stated as: the gap I saw, then the decision it led to.

### 2.1 The model only lives inside the notebook

**What I saw:** the trained model is never saved, and the only "prediction" is a
hard-coded patient at the bottom of the notebook. To make a new prediction you
would have to re-run the whole notebook.

**So I decided to:** persist the trained model and its metadata as versioned
artifacts, and put a real serving layer in front of it, a FastAPI `/predict`
endpoint with typed input validation. The hand-written patient becomes an actual
API request anyone can send.

### 2.2 It cannot explain an individual prediction

**What I saw:** the notebook explains the model *globally* (Random Forest feature
importance), but it cannot say why *this* patient was flagged. In a clinical
setting the per-case reason is exactly what matters.

**So I decided to:** compute SHAP values for every prediction and return the
per-feature contribution and the top risk factors alongside the score. To keep
those explanations in real clinical units, I also kept the model tree-based and
unscaled (the baseline scaled features for its LR/SVM/KNN models; I did not need
to).

### 2.3 It trained on a data leak it did not notice

**What I saw:** the notebook explicitly states it "verified no missing values
were present." That is true for literal blanks, but not for the real problem: in
this dataset a `cholesterol` of `0` is a placeholder for "not measured," and
18.7% of rows have it. Those rows have an 88% disease rate versus 55% overall.
The baseline trained straight through this, so part of what its model learned was
"cholesterol field is blank," an artifact of how the source data was merged, not
a clinical signal.

**So I decided to:** treat `cholesterol == 0` and `resting_bp_s == 0` as missing
and impute them with the training-set median, while deliberately leaving
`oldpeak == 0` alone (there a zero is clinically real). I fit the imputation on
training data only and apply the identical transform at serve time, so there is
no leakage and no train/serve skew. When I retrained, ROC-AUC barely moved
(0.932 to 0.927) and cholesterol fell from the 5th to the 8th most important
feature, which confirmed the model had been leaning on the artifact and now was
not.

### 2.4 It selected models on the wrong metric

**What I saw:** the baseline tuned and compared models on cross-validated
*accuracy*. Accuracy is threshold-dependent and sensitive to the mild class
imbalance here, and a risk model is really a ranking problem.

**So I decided to:** select and tune on cross-validated ROC-AUC instead, which is
threshold-free and imbalance-robust, and only spend the tuning budget on the
model that wins the comparison.

### 2.5 Its probabilities were never calibrated

**What I saw:** the notebook reports a "probability of heart disease" for the
simulated patient but never checks whether that number is actually a calibrated
probability. Random Forest outputs are vote fractions, which rank well but are
not true probabilities.

**So I decided to:** calibrate with Platt scaling (chosen over isotonic because
the sample is small) and, crucially, *measure* it, the Brier score and a
reliability diagram, rather than assert it. ROC-AUC is unchanged because
calibration is monotonic; the score served is now an honest probability. I keep
the raw tree model only for SHAP, since calibration does not change the ranking
SHAP explains.

### 2.6 It is a one-shot evaluation with no life after deployment

**What I saw:** the baseline evaluates once on a test set and stops. A deployed
version of it would have no way to know if the patients it sees later stop
looking like the patients it trained on.

**So I decided to:** log every prediction to a structured store and add Evidently
drift monitoring that compares live inputs against the training baseline, plus a
Streamlit dashboard to see model health. I was careful to scope this honestly:
it detects *input* drift only; detecting *performance* drift would need
ground-truth outcomes I do not collect. I also added a simulate mode so the
detector can be shown actually firing.

### 2.7 It is not reproducible or deployable as software

**What I saw:** no pinned environment, no container, no tests, no CI. It runs on
the author's machine.

**So I decided to:** pin every dependency, containerize the service, and add a
GitHub Actions pipeline that trains the model and runs the test suite on every
push, so the whole thing reproduces from a clean checkout.

---

## 3. What the project is, in one line

The through-line of every decision above: **take a good notebook analysis and
turn it into a small, honest, production-style ML system, fixing a real data
leak and adding the explainability, calibration, monitoring, and reproducibility
the analysis never had.**

Architecture that resulted:

```
Training (src/train.py)      select RF / GB / XGBoost by CV ROC-AUC, tune,
                             calibrate, persist artifacts + metrics
Serving (api/main.py)        FastAPI /predict: calibrated risk score,
                             per-feature SHAP, top risk factors; logs each request
Monitoring (src/monitor.py)  Evidently input-drift vs training baseline
                             (+ a simulate mode that deliberately triggers drift)
Dashboard (dashboard/app.py) model health, predictions, drift status
Docker + GitHub Actions      reproduce the whole pipeline from source
```

`src/preprocessing.py` is shared by training and serving so input handling is
identical in both.

---

## 4. Milestones and status

Each is done only when its acceptance criteria hold.

| # | Milestone | Answers baseline gap | Status |
|---|-----------|----------------------|--------|
| M1 | Reproducible environment, structure, data | 2.7 | Done |
| M2 | Training + model selection on CV ROC-AUC | 2.1, 2.4 | Done |
| M3 | Zero-as-missing data-quality fix | 2.3 | Done |
| M4 | FastAPI serving with SHAP + logging | 2.1, 2.2, 2.6 | Done |
| M5 | Probability calibration (Brier + reliability) | 2.5 | Done |
| M6 | Evidently drift monitoring (+ simulate) | 2.6 | Done |
| M7 | Streamlit dashboard | 2.6 | Done |
| M8 | Docker + GitHub Actions CI | 2.7 | Done |
| M9 | README + model card (with fairness analysis) | all | Done |

Current model: RandomForest, test ROC-AUC ~0.93 (not directly comparable to the
baseline's 0.90, since the feature set, split, and preprocessing differ; the
point was never a higher number, it was a better-engineered system).

---

## 5. What I deliberately did not do, and what is next

**Non-goals for this iteration:** clinical validity (this is educational, not a
medical device), leaderboard-chasing accuracy, and production infrastructure.

**Known limitations I am carrying knowingly:**
- Prediction logging is a local JSONL file: not concurrency-safe, not durable in
  an ephemeral container.
- Monitoring is input drift only; performance drift needs labels I do not collect.
- The data is small and heavily male-skewed, so the female subgroup metrics in
  the model card are not reliable.
- Training runs inside the Docker build and inside CI, which is convenient here
  but not how training and serving should be separated at scale.

**Future work, in priority order:**
1. Replace the JSONL sink with a database or event stream behind the existing
   logging interface.
2. Add a model registry and versioned artifacts; separate training from serving.
3. Add a labelling/feedback loop to enable performance (not just input) drift.
4. Add authentication and rate limiting to the API.
5. Tune the decision threshold to the real cost of a false negative rather than
   defaulting to 0.5.

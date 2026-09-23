# Animal Disease Outbreak Prediction — ML Pipeline

A detailed walkthrough of how the machine learning model was built from scratch to predict animal disease outbreaks across 60 disease categories.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Dataset](#2-the-dataset)
3. [Step 1 — Loading & Column Selection](#step-1--loading--column-selection)
4. [Step 2 — Handling Missing Values](#step-2--handling-missing-values)
5. [Step 3 — Binary Column Cleanup](#step-3--binary-column-cleanup)
6. [Step 4 — Cleaning the Target Column](#step-4--cleaning-the-target-column)
7. [Step 5 — Filtering Rare Diseases](#step-5--filtering-rare-diseases)
8. [Step 6 — Disease Name Consolidation](#step-6--disease-name-consolidation)
9. [Step 7 — Cleaning Species Names](#step-7--cleaning-species-names)
10. [Step 8 — Train / Test Split](#step-8--train--test-split)
11. [Feature Engineering](#feature-engineering)
    - [Step 9 — Species Cardinality Compression](#step-9--species-cardinality-compression)
    - [Step 10 — Country Frequency Encoding](#step-10--country-frequency-encoding)
    - [Step 11 — Cyclical Month Encoding](#step-11--cyclical-month-encoding)
    - [Step 12 — Susceptible Population Log Transform](#step-12--susceptible-population-log-transform)
12. [Step 13 — One-Hot & Scaling Pipeline](#step-13--one-hot--scaling-pipeline)
13. [Step 14 — Label Encoding the Target](#step-14--label-encoding-the-target)
14. [Step 15 — Model Training with 5-Fold Cross-Validation](#step-15--model-training-with-5-fold-cross-validation)
15. [Step 16 — Results & Evaluation](#step-16--results--evaluation)
16. [Step 17 — Saving the Model & Artifacts](#step-17--saving-the-model--artifacts)
17. [Full Feature Summary](#full-feature-summary)

---

## 1. The Problem

The goal is a **multiclass classification** task: given information about an animal outbreak event (which species, where, when, how many animals are at risk, what type of pathogen is suspected), predict **which disease** is most likely causing it.

This is directly useful in veterinary epidemiology — when a new outbreak is reported, the diagnostic process can take days or weeks. A model can instantly suggest the most probable diagnosis based on patterns in historical data.

---

## 2. The Dataset

**Source file:** `ml/dataset.csv`

The raw dataset contains **6,363 rows** and **48 columns**. It is an international disease outbreak reporting dataset covering events from multiple continents over several years.

Most of the 48 columns are administrative metadata (intervention types, event IDs, lab test statuses, slaughter counts, etc.) that are not directly useful for prediction. The meaningful signal lives in just 12 columns.

**Full raw column list includes:**
`event_id`, `country_name`, `iso_code`, `admin_division`, `disease_name`, `disease_subtype`, `disease_group`, `causal_agent_type`, `species_name`, `is_wild`, `epi_unit_type`, `is_domestic`, `is_aquatic`, `latitude`, `longitude`, `outbreak_start_date`, `susceptible`, `cases`, `deaths`, `killed`, `slaughtered`, `intervention_quarantine_applied`, and many more.

---

## Step 1 — Loading & Column Selection

```python
import pandas as pd

df = pd.read_csv('dataset.csv')

# Convert date and derive month + season
df["outbreak_start_date"] = pd.to_datetime(df["outbreak_start_date"])
df["month"] = df["outbreak_start_date"].dt.month_name()

df["season"] = df["outbreak_start_date"].dt.month.map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
})

columns_to_keep = [
    "species_name", "is_wild", "is_domestic", "is_aquatic",
    "causal_agent_type", "country_name", "latitude", "longitude",
    "month", "season", "susceptible", "epi_unit_type", "disease_name"
]

data = df[columns_to_keep]
```

**Why these columns?**

| Column | What it captures |
|---|---|
| `species_name` | Which animal species was affected (Cattle, Swine, Birds, etc.) |
| `is_wild` / `is_domestic` / `is_aquatic` | Whether the animal lives in the wild, is farmed, or is aquatic — affects disease exposure pathways |
| `causal_agent_type` | Broad pathogen class: Virus, Bacterium, Parasite, Prion, Fungus |
| `country_name` | Country where the outbreak happened |
| `latitude` / `longitude` | Precise geographic coordinates |
| `month` | Month the outbreak started (derived from date) |
| `season` | Bioclimatic season (derived from month) |
| `susceptible` | How many animals were at risk in the affected population |
| `epi_unit_type` | Setting where outbreak occurred: Farm, Forest, Apiary, Slaughterhouse, etc. |
| `disease_name` | **Target column** — what disease we are predicting |

**Two new features are derived from the date:**
- **Month** — captures seasonal disease patterns (e.g., Bluetongue surges in summer when midges are active).
- **Season** — a coarser grouping (Winter, Summer, Monsoon, Post-Monsoon) derived from the month number.

---

## Step 2 — Handling Missing Values

```python
data = data.dropna()
print("Remaining rows:", data.shape[0])  # → 6,332 rows
```

Any row that has at least one missing value in the 13 selected columns is dropped entirely. This removes **31 rows** from 6,363, leaving **6,332 clean rows**. The drop rate is minimal (~0.5%), so imputation was not needed.

---

## Step 3 — Binary Column Cleanup

```python
binary_cols = ["is_wild", "is_domestic", "is_aquatic"]
data[binary_cols] = data[binary_cols].astype(int)
```

The three habitat indicator columns (`is_wild`, `is_domestic`, `is_aquatic`) were stored as floats (e.g., `1.0`, `0.0`). They are converted to integers so they behave correctly as binary flags (0 or 1) during model training.

---

## Step 4 — Cleaning the Target Column

```python
data["disease_name"] = (
    data["disease_name"]
    .str.replace(r"\s*\([^)]*\)", "", regex=True)
    .str.strip()
)
```

At this stage, there are **208 unique disease names** in the dataset. Many of them are the same disease written differently in the source data. For example:

- `"African swine fever virus (Inf. with)"` → becomes → `"African swine fever virus"`
- `"Influenza A viruses of high pathogenicity (non-poultry including wild birds) (2017-)"` → becomes → `"Influenza A viruses of high pathogenicity"`

The regex `\s*\([^)]*\)` removes anything inside parentheses and the surrounding spaces. This makes the names consistent before further processing.

---

## Step 5 — Filtering Rare Diseases

```python
disease_counts = data["disease_name"].value_counts()

data_filtered = data[
    data["disease_name"].isin(
        disease_counts[disease_counts >= 10].index
    )
]

print("Original rows:", 6332)
print("Remaining rows:", 5903)
print("Remaining unique diseases:", 78)
```

After removing parenthetical suffixes, there are still 208 unique disease names. Many of these are extremely rare — appearing fewer than 10 times in the entire dataset. A model cannot learn meaningful patterns from only 1–9 examples.

Any disease with **fewer than 10 records is dropped entirely** from the dataset.

**After this filter:**
- Original rows: 6,332
- Remaining rows: 5,903
- Unique diseases: **78** (down from 208)

---

## Step 6 — Disease Name Consolidation

Even after cleaning parentheses, the same biological disease often appears under different names because of different reporting conventions across countries. A manual disease mapping dictionary was created to merge these synonyms:

```python
disease_mapping = {
    "African swine fever virus": "African swine fever",
    "Influenza A viruses of high pathogenicity": "Highly pathogenic avian influenza",
    "High pathogenicity avian influenza viruses": "Highly pathogenic avian influenza",
    "Highly pathogenic influenza A viruses": "Highly pathogenic avian influenza",
    "Foot and mouth disease virus": "Foot and mouth disease",
    "Newcastle disease virus": "Newcastle disease",
    "Rabies virus": "Rabies",
    "Bluetongue virus": "Bluetongue",
    "Lumpy skin disease virus": "Lumpy skin disease",
    "Peste des petits ruminants virus": "Peste des petits ruminants",
    "Rift Valley fever virus": "Rift Valley fever",
    "Equine influenza virus": "Equine influenza",
    "Classical swine fever virus": "Classical swine fever",
    "Viral haemorrhagic septicaemia virus": "Viral haemorrhagic septicaemia",
    "African horse sickness virus": "African horse sickness",
    "Aujeszky's disease virus": "Aujeszky's disease",
    "Koi herpesvirus disease": "Koi herpesvirus",
    "Aethina tumida": "Small hive beetle infestation",
    "Small hive beetle Inf.": "Small hive beetle infestation"
}

data_filtered["disease_name"] = data_filtered["disease_name"].replace(disease_mapping)
```

**Result after consolidation:**
- Unique diseases: **60** (down from 78)
- Minimum samples per disease: 10
- Maximum samples per disease: 1,890 (Highly pathogenic avian influenza)
- Average samples per disease: ~98

For example, the three avian influenza naming variants (`Influenza A viruses of high pathogenicity`, `High pathogenicity avian influenza viruses`, `Highly pathogenic influenza A viruses`) are all correctly merged into the single label `Highly pathogenic avian influenza`, significantly increasing the training signal.

---

## Step 7 — Cleaning Species Names

```python
data_filtered["species_name"] = (
    data_filtered["species_name"]
    .str.replace(r"\s*\([^)]*\)", "", regex=True)
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
```

Species names in the raw data contain noisy formatting — parenthetical taxonomic notes, embedded newline characters, and extra whitespace. For example:

- `"Cattle\n(Bos taurus)"` → `"Cattle"`
- `"Wildlife  (species unspecified)"` → `"Wildlife"`

This is done on the full dataset before splitting so the cleaning is consistent. The same logic is later applied inside the prediction function at inference time.

---

## Step 8 — Train / Test Split

```python
from sklearn.model_selection import train_test_split

X = data_filtered.drop(columns=["disease_name"])
y = data_filtered["disease_name"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))   # → 4,722
print("Testing rows:", len(X_test))     # → 1,181
```

The dataset is split **80% training / 20% testing**.

The important keyword here is `stratify=y`. This ensures that the percentage of records belonging to each disease is preserved in both the train and test sets. Without stratification, a 60-class imbalanced dataset could result in rare diseases being underrepresented in the test set (or missing entirely), which would make evaluation misleading.

**Split result:**
- Training rows: **4,722**
- Testing rows: **1,181**

> ⚠️ All feature engineering steps from this point forward are **fit only on the training set** and then applied to the test set. This prevents data leakage.

---

## Feature Engineering

The 12 raw input features cannot be fed directly to the model as-is. Several transformations are needed to convert them into a numerical, model-friendly format.

---

### Step 9 — Species Cardinality Compression

```python
species_counts = X_train["species_name"].value_counts()
common_species = species_counts[species_counts >= 10].index

X_train["species_name"] = X_train["species_name"].where(
    X_train["species_name"].isin(common_species), "Other"
)

X_test["species_name"] = X_test["species_name"].where(
    X_test["species_name"].isin(common_species), "Other"
)
```

The `species_name` column has hundreds of unique values. Many are extremely rare (e.g., a species appearing only once or twice). When One-Hot Encoding is applied later, a rare species that the model has barely seen cannot contribute meaningful signal — but it would still inflate the feature space.

**Solution:** Any species appearing fewer than 10 times in the **training set** is replaced with the generic label `"Other"`. This compresses the cardinality, reduces noise, and ensures the model only learns from species with enough examples. It also handles unseen species at inference time — a species never seen during training will also become `"Other"`.

---

### Step 10 — Country Frequency Encoding

```python
country_freq = X_train["country_name"].value_counts(normalize=True)

X_train["country_freq"] = X_train["country_name"].map(country_freq)
X_test["country_freq"] = X_test["country_name"].map(country_freq).fillna(0)

X_train = X_train.drop(columns=["country_name"])
X_test = X_test.drop(columns=["country_name"])
```

There are **187 unique countries** in the dataset. Applying One-Hot Encoding to the country would create 187 new binary columns — a massive sparse feature space that's computationally expensive and prone to overfitting.

Instead, **frequency encoding** replaces each country name with a single number: the proportion of training rows that came from that country. For example, if France appears in 8% of training rows, every French record becomes `0.08`. This single continuous number captures how "active" a country is in global disease reporting, which is a meaningful signal.

For countries in the test set that never appeared in training, the frequency is set to `0.0` (`fillna(0)`).

The original `country_name` text column is then dropped — it has been replaced entirely by `country_freq`.

---

### Step 11 — Cyclical Month Encoding

```python
month_map = {
    "January": 1, "February": 2, ..., "December": 12
}

X_train["month_num"] = X_train["month"].map(month_map)
X_train["month_sin"] = np.sin(2 * np.pi * X_train["month_num"] / 12)
X_train["month_cos"] = np.cos(2 * np.pi * X_train["month_num"] / 12)

X_train = X_train.drop(columns=["month", "month_num"])
```

Months have a natural **circular** structure — December (12) and January (1) are only 1 month apart, but if you encode them as integers (12 and 1), the model sees them as 11 units apart. This is wrong.

Cyclical encoding solves this by mapping month numbers onto a circle using **sine and cosine**:
- `month_sin = sin(2π × month / 12)`
- `month_cos = cos(2π × month / 12)`

Together, these two values uniquely and continuously represent any month. January and December are now correctly close to each other in the feature space. This is especially important for diseases that peak around winter transitions (e.g., December–January).

The original `month` text column and the intermediate `month_num` integer are both dropped afterwards.

---

### Step 12 — Susceptible Population Log Transform

```python
X_train["susceptible"] = np.log1p(X_train["susceptible"])
X_test["susceptible"] = np.log1p(X_test["susceptible"])
```

The `susceptible` column (number of animals at risk) is **extremely right-skewed**:
- Minimum: 0
- Median: ~105
- Maximum: 188,081,276

Most outbreaks affect a few hundred animals, but a small number of events involve millions. If fed raw to the model, these extreme values dominate the feature and distort the model's learned representations.

`log1p(x) = log(1 + x)` compresses the range dramatically while preserving the ordering. It also safely handles `0` values (since `log(0)` is undefined, `log(1 + 0) = 0`). After this transformation, the distribution becomes approximately bell-shaped and much more useful for the model.

---

## Step 13 — One-Hot & Scaling Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_features = ["species_name", "causal_agent_type", "season", "epi_unit_type"]

numeric_features = [
    "is_wild", "is_domestic", "is_aquatic",
    "latitude", "longitude", "susceptible",
    "country_freq", "month_sin", "month_cos"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", StandardScaler(), numeric_features)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("Training shape:", X_train_encoded.shape)  # → (4722, 73)
print("Testing shape:",  X_test_encoded.shape)   # → (1181, 73)
```

A `ColumnTransformer` applies two different transformations simultaneously to different groups of columns:

**Categorical columns → One-Hot Encoding**

`species_name`, `causal_agent_type`, `season`, and `epi_unit_type` are text categories. One-Hot Encoding converts them into a grid of binary columns (one column per unique category value). `handle_unknown="ignore"` means that if a new category appears at inference time that was never seen during training, it simply gets a row of zeros — no error is thrown.

**Numeric columns → Standard Scaling**

The 9 numeric features (`is_wild`, `is_domestic`, `is_aquatic`, `latitude`, `longitude`, `susceptible`, `country_freq`, `month_sin`, `month_cos`) are rescaled to have **zero mean and unit variance**. This ensures that no single numeric feature dominates others purely because of its magnitude (e.g., latitude ranges from −90 to +90, while `is_wild` is only 0 or 1).

**Critical rule:** The `preprocessor` is `.fit_transform()` on training data only. The test set is transformed with `.transform()` only — using the vocabulary and scaling parameters learned from training. This avoids data leakage.

**Result: 73 total features** after all encoding and transformations.

---

## Step 14 — Label Encoding the Target

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
```

CatBoost (and most ML libraries) expects the target to be **integers**, not text strings. `LabelEncoder` converts the 60 disease name strings into integer class indices (0 through 59) alphabetically.

```
0  = African horse sickness
1  = African swine fever
2  = American foulbrood of honey bees
3  = Anthrax
...
59 = White spot disease
```

The `label_encoder` object is saved alongside the model so it can reverse the mapping at prediction time — converting a predicted integer back to a human-readable disease name.

---

## Step 15 — Model Training with 5-Fold Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(X_train_encoded, y_train_encoded), start=1):

    X_fold_train = X_train_encoded[train_index]
    X_fold_val   = X_train_encoded[val_index]
    y_fold_train = y_train_encoded[train_index]
    y_fold_val   = y_train_encoded[val_index]

    fold_model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.09,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        auto_class_weights="Balanced",
        random_seed=42,
        task_type="GPU",
        devices="0",
        verbose=False
    )

    fold_model.fit(X_fold_train, y_fold_train)
```

### Why Cross-Validation?

With only ~4,700 training records split across 60 classes, a single train/validation split could produce a misleadingly good or bad estimate. **5-Fold Stratified Cross-Validation** gives a much more reliable performance estimate by training 5 separate models, each on a different 80%/20% split of the training data, and averaging the scores.

`StratifiedKFold` is used instead of regular `KFold` to ensure each fold's validation split maintains the same class distribution as the full training set. This is critical when classes are imbalanced.

### CatBoost Hyperparameters Explained

| Parameter | Value | Why |
|---|---|---|
| `iterations` | 1000 | Number of gradient boosting trees to grow |
| `depth` | 8 | Maximum depth of each tree — deeper trees learn more complex patterns |
| `learning_rate` | 0.09 | How much each new tree corrects the previous residual — lower is more careful |
| `loss_function` | `MultiClass` | Appropriate loss for 60-class classification |
| `eval_metric` | `TotalF1` | Optimizes for the harmonic mean of precision and recall across all classes |
| `auto_class_weights` | `Balanced` | Automatically upweights rare classes to prevent the model from ignoring minority diseases |
| `task_type` | `GPU` | Trains on GPU (dramatically faster with 1000 trees and 73 features) |

### Why `auto_class_weights="Balanced"`?

The dataset is **highly imbalanced**:
- Highly pathogenic avian influenza: 1,890 records
- Many rare diseases: only 10 records each

Without class balancing, the model would almost always predict the dominant class and still achieve decent accuracy. `Balanced` weighting assigns each class a weight inversely proportional to its frequency, forcing the model to pay equal attention to rare diseases.

---

## Step 16 — Results & Evaluation

```
===== 5-Fold Cross-Validation Results =====

Fold 1: Accuracy 0.7704 | Macro F1 0.6042 | Weighted F1 0.7685
Fold 2: Accuracy 0.7735 | Macro F1 0.6028 | Weighted F1 0.7677
Fold 3: Accuracy 0.7860 | Macro F1 0.6200 | Weighted F1 0.7828
Fold 4: Accuracy 0.7701 | Macro F1 0.5708 | Weighted F1 0.7675
Fold 5: Accuracy 0.7828 | Macro F1 0.6130 | Weighted F1 0.7801

Average Accuracy:      0.7766  (±0.0066)
Average Macro F1:      0.6021  (±0.0169)
Average Weighted F1:   0.7734  (±0.0067)
```

### What the metrics mean

**Accuracy (77.7%):** Out of every 100 outbreak records, the model correctly predicts the disease in ~78 of them. This is a strong result for a 60-class problem.

**Macro F1 (60.2%):** The average F1 score calculated independently for each of the 60 disease classes, then averaged equally. This treats all classes as equally important — so a rare disease with only 10 examples counts as much as the most common one. The lower Macro F1 compared to Weighted F1 reflects the genuine challenge of predicting rare diseases with limited examples.

**Weighted F1 (77.3%):** The average F1 score weighted by each class's share of the dataset. This is closer to Accuracy since common diseases influence it more, and is a good overall performance indicator.

**Low standard deviation (±0.0066):** The very small variation across the 5 folds confirms the model is stable and not sensitive to which subset of data it trains on — a sign that the model has genuinely learned the underlying patterns.

---

## Step 17 — Saving the Model & Artifacts

```python
import joblib

# Save the CatBoost model
fold_model.save_model("saved_models/catboost_final.cbm")

# Save the disease label encoder
joblib.dump(label_encoder, "saved_models/label_encoder.pkl")

# Save preprocessing components needed at inference time
joblib.dump({
    "preprocessor": preprocessor,      # ColumnTransformer (fitted)
    "country_freq": country_freq,      # Training country frequency table
    "common_species": common_species   # Set of species with >= 10 occurrences
}, "saved_models/preprocessing_assets.pkl")
```

Three artefacts are saved. All three are required to make a prediction:

| File | Contents | Used for |
|---|---|---|
| `catboost_final.cbm` | Trained CatBoost model weights (125 MB) | Running the actual prediction |
| `label_encoder.pkl` | Maps integer predictions back to disease names | Converting model output to human-readable labels |
| `preprocessing_assets.pkl` | The fitted `ColumnTransformer`, the `country_freq` mapping, and the `common_species` set | Applying the exact same feature transformations to new data at inference time |

> **Why save the preprocessing objects?** The `StandardScaler` inside the `ColumnTransformer` learned the mean and standard deviation of the training data. When a new outbreak record arrives, it must be scaled using **those same training statistics** — not recalculated from scratch. Similarly, `country_freq` must come from the training distribution, and `common_species` must reflect what was seen during training. Using these saved objects guarantees identical preprocessing between training and inference.

---

## Full Feature Summary

| # | Input Feature | Type | Transformation Applied |
|---|---|---|---|
| 1 | `species_name` | Categorical | Rare species → "Other", then One-Hot Encoded |
| 2 | `is_wild` | Binary (0/1) | Standard Scaled |
| 3 | `is_domestic` | Binary (0/1) | Standard Scaled |
| 4 | `is_aquatic` | Binary (0/1) | Standard Scaled |
| 5 | `causal_agent_type` | Categorical | One-Hot Encoded |
| 6 | `country_name` | Categorical (187 values) | Replaced with frequency proportion, then Standard Scaled |
| 7 | `latitude` | Continuous | Standard Scaled |
| 8 | `longitude` | Continuous | Standard Scaled |
| 9 | `month` | Ordinal / Cyclic | Replaced with `month_sin` and `month_cos` (2 features), then Standard Scaled |
| 10 | `season` | Categorical | One-Hot Encoded |
| 11 | `susceptible` | Continuous (heavy skew) | log1p transformed, then Standard Scaled |
| 12 | `epi_unit_type` | Categorical | One-Hot Encoded |

**12 raw input features → 73 encoded model features**

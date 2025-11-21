# Car Damage Claim Estimator 🚗🛠️

This project is an **end-to-end car damage claim estimator** that takes a **damaged car image + basic vehicle details** and predicts:

- **Damage type** (scratch, dent, glass shatter, etc.)
- **Estimated repair cost**

It combines **deep image features**, **classical machine learning**, and **big data tools (Spark)** into a single pipeline, and exposes a simple **web UI** using Gradio.

---

## 🔍 What this project does

- Loads a public **car damage dataset** (images + COCO annotations)
- Uses a pre-trained **EfficientNetB0** model to turn each image into a **numeric feature vector**
- Trains:
  - a **KNN classifier** to predict `damage_type`
  - a **Spark GBTRegressor** to predict `repair_cost`
- Stores the engineered data in **Parquet** for efficient reuse
- Provides a **Gradio web interface** to upload an image and get:
  - predicted damage type  
  - severity score  
  - estimated repair cost  

---

## 🧠 Model & Approach

### 1. Dataset & Labels

- Dataset: **CARDD – Car Damage Detection Dataset** from Kaggle
- Images: damaged cars from different angles
- Annotations: **COCO-style JSON** with image metadata and damage labels
- I parse the JSON and assign a **main `damage_type`** (e.g., scratch, dent, glass_shatter) to each image.

### 2. Image Feature Extraction (EfficientNetB0)

- I use **EfficientNetB0** (pre-trained on ImageNet) as a **feature extractor**.
- For each image, EfficientNetB0 outputs a **vector of numbers** (an embedding) that represents its visual content.
- These vectors become the input features for downstream models.

**Why:**  
I don’t have millions of images or huge GPUs to train a CNN from scratch. Using EfficientNetB0 gives me strong visual features quickly.

### 3. Damage Type Classification (KNN)

- Model: **KNeighborsClassifier** (scikit-learn)
- Input: EfficientNetB0 image embeddings
- Output: `damage_type`

**Intuition:**  
KNN finds the **nearest images in feature space** and predicts the damage type based on their labels.

### 4. Repair Cost Regression (Spark GBTRegressor)

- I generate a **synthetic `repair_cost`** using:
  - base cost per damage type  
  - **severity_score** (0.5–5.0)  
  - **car age** (from `car_year`)  
  - **mileage**
- I train a **Gradient Boosted Trees regressor (GBTRegressor)** in Spark ML.
- Features include:
  - driver_age  
  - car_year / car_age  
  - mileage  
  - severity_score  
  - encoded `damage_type`  
  - image feature vector

**Why:**  
Gradient Boosted Trees handle **non-linear relationships** well, and Spark ML lets the same code scale to much larger datasets.

---

## 🧱 Tech Stack

- **Language:** Python
- **Big Data & ML:**  
  - Apache **Spark** (PySpark, Spark ML)  
  - **GBTRegressor** for cost prediction
- **Deep Learning:**  
  - **TensorFlow / Keras**  
  - **EfficientNetB0** (feature extractor)
- **Classical ML:**  
  - **scikit-learn** (KNN classifier)
- **Storage:**  
  - **Parquet** (columnar format, Spark/HDFS-friendly)
- **UI:**  
  - **Gradio** (web interface for live predictions)

---

## 🏗️ Pipeline Overview

1. **Ingestion**
   - Download CARDD dataset from Kaggle.
   - Parse COCO JSON to map each image to a main `damage_type`.

2. **Feature Extraction & Claim Table**
   - Run each image through **EfficientNetB0** to get an embedding.
   - Add synthetic metadata: driver_age, car_year, mileage, severity_score.
   - Build a claim-like table: one row per potential claim.

3. **Storage**
   - Save the engineered data as a **Spark DataFrame**.
   - Persist it as **Parquet** for efficient reloading.

4. **Modeling**
   - Train **KNN** (scikit-learn) on embeddings to predict `damage_type`.
   - Train **Spark GBTRegressor** on combined features to predict `repair_cost`.
   - Save the Spark model as a **PipelineModel** (ready for deployment).

5. **Serving**
   - Build a **Gradio interface**:
     - Inputs: car image, driver_age, car_year, mileage
     - Outputs: predicted damage type, severity score, estimated repair cost

---

## ⚙️ How to Run (high level)

1. Open the main notebook (`603Project_source_code.ipynb`) in Jupyter / Colab.
2. Run cells in order:
   - Dataset setup & COCO parsing  
   - EfficientNet feature extraction & Parquet save  
   - GBTRegressor training  
   - KNN training  
   - Final hybrid prediction test  
   - Gradio app block
3. Launch the Gradio interface and try your own test images.

---

## 🚧 Limitations & Future Work

**Current limitations:**

- `repair_cost` is **synthetic** (formula-based), not from real invoices.
- Dataset size is smaller than real insurance-scale data.
- No dedicated fairness or explainability analysis yet.

**Possible future improvements:**

- Use real claim and repair data for training.
- Fine-tune EfficientNet on car damage specifically.
- Add streaming (Kafka + Spark Structured Streaming).
- Deploy the model as an API behind the Gradio UI.

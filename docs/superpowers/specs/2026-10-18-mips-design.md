# Design Specification: MIPS (Model for Invasive Plant Species Detection)

**Date**: 2026-10-18  
**Project**: MIPS (Model for Invasive Plant Species detection)  
**Target Users**: First Lego League (FLL) Team (Ages 9–11) & Coach  
**Status**: Approved for Implementation Planning  

---

## 1. Executive Summary & Goals

MIPS is an evaluation platform and experiment tracking leaderboard designed for a First Lego League (FLL) robotics and innovation project. A team of students (ages 9–11) will use Google Teachable Machine to build and iteratively improve image classification models to detect invasive plant species.

### Primary Objectives:
1. **Model Ingestion & Evaluation**: Allow students to export their trained models from Google Teachable Machine (as standard Keras `.h5` or TFLite `.zip` bundles), upload them to MIPS, and have them automatically evaluated against an unseen, locked benchmark test dataset.
2. **Pedagogical Machine Learning**: Teach students authentic ML principles—moving beyond raw accuracy to understand the **Confusion Matrix (TP, FP, TN, FN)**, **Precision**, **Recall**, and **F1-Score**, showing them how to use error analysis to guide their next training experiment ("hill-climbing").
3. **Multi-Board Leaderboards**:
   - **Board 1**: Stage 1 Classifier (`plant` vs. `not_plant` noise filter).
   - **Board 2**: Stage 2 Classifier (`invasive` vs. `non_invasive` botanical detector).
   - **Board 3**: End-to-End Field Robot Pipeline (evaluates Stage 1 $\rightarrow$ Stage 2 in sequence).
4. **Dataset Management & Storage Efficiency**: Provide a coach admin tool to intake raw student photos, auto-downsample them to ML-ready resolutions ($448 \times 448$) to keep storage featherweight (< 25 MB), create stratified 70/30 train/test splits, and export clean training packs for Teachable Machine.
5. **Industry Benchmark (P2)**: A coach-built reference entry using a Vision-Language Model (Gemini 2.5 Flash with System Instructions) pinned on the leaderboard to demonstrate how hand-trained edge models compare to large foundation models.

---

## 2. Feature Prioritization

* **P0 (Core MVP)**:
  * Teachable Machine Model Ingestion (`.zip` containing `keras_model.h5` or `model.tflite` + `labels.txt`).
  * Evaluation Engine (image preprocessing, model inference, metric calculations).
  * True ML Metrics display: $2 \times 2$ Confusion Matrix, Accuracy, Precision, Recall, F1-Score.
  * Interactive Leaderboards for **Board 1** (`plant` vs. `not_plant`) and **Board 2** (`invasive` vs. `non_invasive`).
  * Error Analysis Inspector ("Bloopers Gallery" showing False Positives and False Negatives with model confidence).
  * Local JSON persistent storage (`data/leaderboard_history.json`).
* **P1 (Dataset Admin & Pipeline Simulation)**:
  * Coach Dataset Management tab: image validation, deduplication, class balancing, auto-downsampling, and stratified 70/30 train/test split.
  * Automatic generation of downloadable student training packs (`stage1_train.zip`, `stage2_train.zip`).
  * **Board 3 (End-to-End Pipeline)**: Simulates field robot deployment by cascading Model 1 and Model 2 on a combined test suite.
  * Experiment Progress Timeline chart (visualizing F1-score climbing over successive weeks/sprints).
* **P2 (Advanced Coach Benchmark)**:
  * Gemini 2.5 Flash VLM Benchmark runner using structured System Instructions (SI).
  * Pinned benchmark row on Board 1 and Board 2 to show students how their specialized models compare to a multi-billion parameter foundation model.

---

## 3. System Architecture & Components

```
mips/
├── app.py                      # Main Streamlit web application & navigation
├── core/
│   ├── evaluator.py            # Model extraction, preprocessing, inference & metrics computation
│   ├── dataset_manager.py      # Image validation, resizing, stratified 70/30 splitting, zip packaging
│   ├── gemini_benchmark.py     # (P2) Gemini API client with System Instructions (cached benchmark)
│   └── storage.py              # Persistent read/write operations for leaderboard runs & schema validation
├── data/
│   ├── raw/                    # Raw photo intake organized by stage and class
│   │   ├── stage1/
│   │   │   ├── plant/
│   │   │   └── not_plant/
│   │   └── stage2/
│   │       ├── invasive/
│   │       └── non_invasive/
│   ├── test_dataset/           # Locked 30% test images (never exposed to students)
│   ├── training_packs/         # Generated .zip packs for student training
│   ├── leaderboard_history.json# Persistent log of all evaluations, runs, and metrics
│   └── gemini_cache.json       # (P2) Cached predictions from Gemini VLM to avoid API quota waste
├── tests/                      # Unit tests for core logic
│   ├── test_evaluator.py
│   ├── test_dataset_manager.py
│   └── test_storage.py
├── requirements.txt            # Python dependencies
└── README.md                   # Coach setup guide & student onboarding instructions
```

### Component Responsibilities:

1. **`app.py` (Streamlit Frontend)**:
   * Multi-tab interface:
     - 🌿 **Board 1 (Plant vs. Not-Plant)**
     - ⚠️ **Board 2 (Invasive vs. Non-Invasive)**
     - 🏆 **Board 3 (Full Pipeline Simulation)**
     - 📤 **Submit Model**
     - 🛠️ **Coach Dataset Admin** (PIN-protected)
   * Renders metric cards, sortable leaderboard tables, interactive Plotly confusion matrices, and the error inspector.

2. **`core/evaluator.py`**:
   * Accepts an uploaded `.zip` file from Google Teachable Machine.
   * Detects and loads either `keras_model.h5` or `model.tflite` along with `labels.txt`.
   * Standardizes incoming images: loads, converts to RGB, resizes to $224 \times 224$, normalizes to $[-1, 1]$.
   * Computes binary metrics:
     $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
     $$\text{Precision} = \frac{TP}{TP + FP}$$
     $$\text{Recall} = \frac{TP}{TP + FN}$$
     $$\text{F1-Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   * Identifies False Positives (FP) and False Negatives (FN) ranked by prediction confidence for diagnostic review.

3. **`core/dataset_manager.py`**:
   * Scans incoming images in `data/raw/`.
   * Validates file integrity (skips corrupt files, unsupported formats).
   * Downsamples images to a maximum dimension of $448 \times 448$ at 85% JPEG quality to ensure minimal disk footprint (< 30 KB per image).
   * Performs stratified random sampling (70% Train, 30% Test) maintaining class balance.
   * Generates student training bundles (`stage1_train.zip`, `stage2_train.zip`) and locks test images into `data/test_dataset/`.

4. **`core/storage.py`**:
   * Manages `data/leaderboard_history.json`.
   * Schema for each run record:
     ```json
     {
       "id": "run-uuid-1234",
       "timestamp": "2026-10-18T14:30:00Z",
       "stage": "stage1",
       "team_name": "Team Kudzu",
       "model_version": "v2.1",
       "notes": "Added 30 photos with harsh shadows and sun glare",
       "metrics": {
         "accuracy": 0.88,
         "precision": 0.85,
         "recall": 0.92,
         "f1": 0.883,
         "tp": 23,
         "fp": 4,
         "tn": 21,
         "fn": 2
       },
       "errors": [
         {
           "filename": "test_042.jpg",
           "true_label": "not_plant",
           "predicted_label": "plant",
           "confidence": 0.89
         }
       ]
     }
     ```

5. **`core/gemini_benchmark.py` (P2 - Coach Built)**:
   * Uses Google GenAI SDK (`google-genai`) with model `gemini-2.5-flash`.
   * Evaluates test images zero-shot using precise botanical System Instructions:
     - **Stage 1 SI**: *"You are an expert botanical image classifier. Identify whether the given image contains a plant or plant material, or if it is a non-plant object (e.g. tools, shoes, ground, rocks, artificial items). Respond in strict JSON: {'label': 'plant' | 'not_plant', 'confidence': float, 'reasoning': string}."*
     - **Stage 2 SI**: *"You are an expert invasive species botanist. Identify whether the plant shown in the image is an invasive plant species or a non-invasive (native/harmless) plant. Respond in strict JSON: {'label': 'invasive' | 'non_invasive', 'confidence': float, 'reasoning': string}."*
   * Saves evaluation outputs to `data/gemini_cache.json` and injects benchmark rows into the leaderboards.

---

## 4. User Interaction & Workflow

### Student Workflow:
1. Student team downloads `stage1_train.zip` or `stage2_train.zip` from MIPS.
2. Students open [Google Teachable Machine (Image Project)](https://teachablemachine.withgoogle.com/).
3. They drag the training photos into the respective classes (`plant` / `not_plant` or `invasive` / `non_invasive`).
4. They click **Train Model**.
5. They click **Export Model** $\rightarrow$ **TensorFlow** $\rightarrow$ **Keras** (or **TensorFlow Lite**) $\rightarrow$ **Download my model**.
6. In MIPS, they navigate to the **Submit Model** tab, enter their Team Name, Version, and Experiment Notes, drop their exported `.zip` file, and click **Run Evaluation**.
7. Within 5 seconds, MIPS displays their Confusion Matrix, F1-Score, and position on the Leaderboard.
8. Students inspect their "Bloopers" to understand their False Negatives and False Positives, forming a hypothesis for their next training round.

### Coach Workflow:
1. Coach collects field photos from students and drops them into `data/raw/`.
2. Coach opens the **Dataset Admin** tab, enters the admin PIN, and inspects class counts.
3. Coach clicks **Process & Generate Splits**. The tool resizes images, produces locked test sets, and outputs student training zips.
4. (P2) Coach clicks **Run Gemini Benchmark** once to establish the foundation model baseline on the leaderboard.

---

## 5. Deployment & Technical Specifications

* **Runtime**: Python 3.10+
* **Core Libraries**:
  * `streamlit`: Web application framework.
  * `tensorflow` (or `tflite-runtime`): Model loading and inference.
  * `pillow`: Image processing and downsampling.
  * `plotly`: Interactive confusion matrices and hill-climbing line charts.
  * `pandas`, `numpy`, `scikit-learn`: Data handling and metric calculations.
  * `google-genai`: (P2) Gemini API SDK.
* **Target Platforms**:
  1. **Local Coach Machine**: Runs `streamlit run app.py` on Mac/PC. Team connects over local Wi-Fi.
  2. **Hugging Face Spaces**: Recommended free cloud hosting (16 GB RAM, 50 GB persistent storage, zero OOM risk).
  3. **Streamlit Community Cloud**: Supported alternative.

---

## 6. Verification & Testing Strategy

1. **Unit Testing**:
   * Evaluator test: mock model loading, ensure metric calculations match manual formulas for known $2 \times 2$ matrices.
   * Dataset manager test: verify that images are correctly resized, splits are mutually exclusive and collectively exhaustive, and training zips are valid.
   * Storage test: verify JSON read/write, schema validation, and score ranking.
2. **End-to-End Integration Test**:
   * Create a synthetic mini-test dataset with 10 dummy images (5 positive, 5 negative).
   * Evaluate a sample exported model to verify the entire pipeline from file upload to leaderboard update.

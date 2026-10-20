# MIPS: Model for Invasive Plant Species Detection

**MIPS** is an ML evaluation platform, dataset manager, and experiment tracking leaderboard designed for First Lego League (FLL) robotics and innovation projects. 

Students (ages 9–11) build and iterate on image classification models using [Google Teachable Machine](https://teachablemachine.withgoogle.com/) and hill-climb their models on an unseen locked test dataset.

---

## 🎯 Architecture & Mission
1. **Stage 1 Filter**: `plant` vs. `not_plant` (Filters out shoes, rocks, soil, and tools).
2. **Stage 2 Classifier**: `invasive` vs. `non_invasive` (Detects target invasive weeds).
3. **Stage 3 Field Robot Simulation**: Combines Stage 1 $\rightarrow$ Stage 2 in sequence to simulate an autonomous Lego robot traversing the field.

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`. (On team Wi-Fi, other students can connect via your local IP, e.g. `http://coach-laptop.local:8501`).

---

## 🧑‍🎓 Student Workflow (Ages 9–11)
1. Download `stage1_train.zip` or `stage2_train.zip` from MIPS.
2. Open [Google Teachable Machine (Image Project)](https://teachablemachine.withgoogle.com/).
3. Drag training photos into the two classes.
4. Click **Train Model**.
5. Click **Export Model** $\rightarrow$ **TensorFlow** $\rightarrow$ **Keras** (or **TensorFlow Lite**) $\rightarrow$ **Download my model**.
6. In MIPS, open the **Submit Model** tab, enter your Team Name & experiment notes, drop the exported `.zip` file, and see your rank on the Leaderboard!
7. Study the **Confusion Matrix** and **Bloopers Gallery** to see what mistakes the model made.

---

## 👩‍🏫 Coach Dataset Administration
- Collect student photos and drop them into `data/raw/stage1/` and `data/raw/stage2/`.
- Open the **Coach Dataset Admin** tab in the app to inspect balance, downsample to $448 \times 448$ (saving disk space), and generate the locked 30% test set and 70% student training packs.

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

# MIPS: Model for Invasive Plant Species Detection

**MIPS** is an ML evaluation platform, dataset manager, and experiment tracking leaderboard designed for First Lego League (FLL) robotics and innovation projects. 

Students (ages 9–11) build and iterate on image classification models using [Google Teachable Machine](https://teachablemachine.withgoogle.com/) and hill-climb their models on an unseen locked test dataset.

---

## 🎯 Architecture & Boards
0. **🐱 Tutorial Board**: `cat` vs. `not_cat` (Onboarding "Hello World" using Hugging Face's `cats_vs_dogs` dataset).
1. **🌿 Stage 1 Filter**: `plant` vs. `not_plant` (Filters out shoes, rocks, soil, and tools).
2. **⚠️ Stage 2 Classifier**: `invasive` vs. `non_invasive` (Detects target invasive weeds).
3. **🏆 Stage 3 Field Robot Simulation**: Combines Stage 1 $\rightarrow$ Stage 2 in sequence to simulate an autonomous Lego robot traversing the field.

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
1. Download a training pack (`intro_train.zip`, `stage1_train.zip`, or `stage2_train.zip`) from MIPS.
2. Open [Google Teachable Machine (Image Project)](https://teachablemachine.withgoogle.com/).
3. Drag training photos into the two classes.
4. Click **Train Model**.
5. Click **Export Model** $\rightarrow$ **TensorFlow** $\rightarrow$ **Keras** (or **TensorFlow Lite**) $\rightarrow$ **Download my model**.
6. In MIPS, open the **Submit Model** tab, enter your Team Name & experiment notes, drop the exported `.zip` file, and see your rank on the Leaderboard!
7. Study the **Confusion Matrix** and **Bloopers Gallery** to see what mistakes the model made.

---

## 👩‍🏫 Coach Dataset Administration
- Open the **Coach Dataset Admin** tab in the app (PIN: `fll2026`).
- Click **⚡ Fetch Cat vs. Not-Cat from Hugging Face** to immediately pull the tutorial starter set.
- Drop team-collected weed photos into `data/raw/stage1/` and `data/raw/stage2/`.
- Click **Process & Generate Splits** to automatically downsample to $448 \times 448$ (saving disk space), and generate the locked 30% test set and 70% student training packs.

---

## ☁️ Deploying to Streamlit Community Cloud

Deploy for free in under 60 seconds with your GitHub account:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** $\rightarrow$ **I already have an app**.
3. Fill in:
   - **Repository**: `terratron-fll/mips`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `terratron-mips` (or your choice)
4. Click **Deploy!**

Your live web platform will be accessible to your students, coaches, and FLL judges at:  
`https://terratron-mips.streamlit.app`

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

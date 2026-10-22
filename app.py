import os
from datetime import datetime
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from core.dataset_manager import DatasetManager
from core.evaluator import ModelEvaluator
from core.gemini_benchmark import GeminiBenchmarkRunner
from core.pipeline import PipelineEvaluator
from core.storage import StorageManager
from scripts.seed_intro_data import seed_cat_dataset

st.set_page_config(
    page_title="MIPS: Invasive Plants ML Leaderboard",
    page_icon="🌿",
    layout="wide",
)

def render_confusion_matrix(metrics: dict, pos_label: str, neg_label: str):
    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    tn = metrics.get("tn", 0)

    z = [[tn, fp], [fn, tp]]
    x = [f"Pred: {neg_label}", f"Pred: {pos_label}"]
    y = [f"Actual: {neg_label}", f"Actual: {pos_label}"]
    annotations = [
        [f"<b>TN</b><br>{tn}", f"<b>FP (False Alarm)</b><br>{fp}"],
        [f"<b>FN (Missed Weed)</b><br>{fn}", f"<b>TP (Caught)</b><br>{tp}"],
    ]

    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=annotations,
        colorscale="Greens",
        showscale=False,
    )
    fig.update_layout(
        title="2x2 Confusion Matrix",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

def render_board(storage: StorageManager, stage: str, stage_name: str, pos_label: str, neg_label: str):
    st.header(f"{stage_name} Leaderboard")
    leaderboard = storage.get_leaderboard(stage=stage)

    # Check for cached Gemini VLM benchmark
    gemini_runner = GeminiBenchmarkRunner()
    cache = gemini_runner.load_cache()
    gemini_entry = cache.get(stage)

    if not leaderboard and not gemini_entry:
        st.info(f"No models submitted to {stage_name} yet. Be the first team to submit!")
        return

    # Top summary metrics
    if leaderboard:
        top_run = leaderboard[0]
        top_metrics = top_run.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🥇 Top Student F1", f"{top_metrics.get('f1', 0.0):.3f}")
        col2.metric("🎯 Top Accuracy", f"{top_metrics.get('accuracy', 0.0) * 100:.1f}%")
        col3.metric("🏆 Top Team", f"{top_run.get('team_name', 'Unknown')} ({top_run.get('model_version', '')})")
        col4.metric("📊 Total Submissions", len(leaderboard))

    # Leaderboard table
    rows = []
    if gemini_entry:
        gm = gemini_entry.get("metrics", {})
        rows.append({
            "Rank": "🤖 BENCHMARK",
            "Team": "Gemini 2.5 Flash",
            "Version": "Zero-Shot VLM (SI)",
            "F1-Score": f"{gm.get('f1', 0.0):.3f}",
            "Accuracy": f"{gm.get('accuracy', 0.0) * 100:.1f}%",
            "Precision": f"{gm.get('precision', 0.0):.3f}",
            "Recall": f"{gm.get('recall', 0.0):.3f}",
            "TP": gm.get("tp", 0),
            "FP": gm.get("fp", 0),
            "TN": gm.get("tn", 0),
            "FN": gm.get("fn", 0),
            "Submitted": "Coach Baseline",
            "Notes": "Google Foundation Model Reference",
        })

    for rank, run in enumerate(leaderboard, 1):
        m = run.get("metrics", {})
        dt = run.get("timestamp", "")[:16].replace("T", " ")
        rows.append({
            "Rank": f"#{rank}",
            "Team": run.get("team_name", ""),
            "Version": run.get("model_version", ""),
            "F1-Score": f"{m.get('f1', 0.0):.3f}",
            "Accuracy": f"{m.get('accuracy', 0.0) * 100:.1f}%",
            "Precision": f"{m.get('precision', 0.0):.3f}",
            "Recall": f"{m.get('recall', 0.0):.3f}",
            "TP": m.get("tp", 0),
            "FP": m.get("fp", 0),
            "TN": m.get("tn", 0),
            "FN": m.get("fn", 0),
            "Submitted": dt,
            "Notes": run.get("notes", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Hill-climbing line chart
    st.subheader("📈 Hill-Climbing Progress Over Time")
    history = storage.get_history(stage=stage)
    chart_data = []
    for h in history:
        m = h.get("metrics", {})
        chart_data.append({
            "Submission": h.get("timestamp", "")[:16].replace("T", " "),
            "Team": h.get("team_name", "Team"),
            "F1-Score": m.get("f1", 0.0),
            "Accuracy": m.get("accuracy", 0.0),
        })

    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        fig_line = px.line(
            chart_df,
            x="Submission",
            y="F1-Score",
            color="Team",
            markers=True,
            title="Team F1-Score Progress",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Detailed inspection of top run
    if leaderboard:
        top_run = leaderboard[0]
        st.subheader(f"🔍 Inspecting Rank #1: {top_run.get('team_name')} ({top_run.get('model_version')})")
        col_cm, col_err = st.columns([1, 1])
        with col_cm:
            render_confusion_matrix(top_run.get("metrics", {}), pos_label=pos_label, neg_label=neg_label)

        with col_err:
            st.write("<b>Bloopers Gallery (Top False Alarms & Missed Weeds):</b>", unsafe_allow_html=True)
            errors = top_run.get("errors", [])
            if not errors:
                st.success("🎉 Perfect score! No errors found on the test dataset.")
            else:
                for err in errors[:3]:
                    err_path = err.get("image_path")
                    caption = f"{err.get('error_type')}: True='{err.get('true_label')}', Pred='{err.get('predicted_label')}' (Conf: {err.get('confidence')})"
                    if err_path and os.path.exists(err_path):
                        st.image(err_path, caption=caption, width=200)
                    else:
                        st.write(f"- {caption} ({err.get('filename')})")

def render_submit_tab(storage: StorageManager, evaluator: ModelEvaluator):
    st.header("📤 Submit Your Teachable Machine Model")

    with st.expander("❓ Exactly Which Export Option to Pick in Google Teachable Machine", expanded=True):
        st.markdown(
            """
            Google Teachable Machine has three export tabs (**TensorFlow.js**, **TensorFlow**, and **TensorFlow Lite**).  
            **Pick ONE of these two supported options:**

            * 🥇 **Option 1 (Recommended): TensorFlow $\\rightarrow$ Keras**
              * Click the **TensorFlow** tab.
              * Select the **Keras** radio button.
              * Click **Download my model** (downloads a `.zip` containing `keras_model.h5` and `labels.txt`).
            
            * 🥈 **Option 2 (Lightweight / Fast): TensorFlow Lite $\\rightarrow$ Floating point**
              * Click the **TensorFlow Lite** tab.
              * Select the **Floating point** radio button.
              * Click **Download my model** (downloads a `.zip` containing `model_unquant.tflite` and `labels.txt`).

            ---
            **❌ What NOT to pick (and why):**
            * ❌ **TensorFlow.js**: This exports a web `model.json` bundle designed to run inside JavaScript browser apps, not Python evaluation backends.
            * ❌ **TensorFlow $\\rightarrow$ SavedModel**: Creates a multi-file folder directory (`saved_model.pb`) rather than a single portable model file.
            * ❌ **TensorFlow Lite $\\rightarrow$ EdgeTPU**: Only runs on specialized Google Coral microchips; fails on standard computer processors.
            * ⚠️ **TensorFlow Lite $\\rightarrow$ Quantized**: Supported, but slightly lower accuracy than Floating point.
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("Team Name", placeholder="e.g., Team Kudzu Hunters")
        model_version = st.text_input("Model Version", placeholder="e.g., v1.2")
    with col2:
        stage = st.selectbox(
            "Which board are you submitting to?",
            options=["intro", "stage1", "stage2"],
            format_func=lambda x: {
                "intro": "🐱 Tutorial: Cat vs Not-Cat",
                "stage1": "🌿 Stage 1: Plant vs Not-Plant",
                "stage2": "⚠️ Stage 2: Invasive vs Non-Invasive",
            }[x],
        )
        notes = st.text_input("Experiment Notes (What did you change?)", placeholder="e.g., Added 20 photos with sun glare and shadows")

    uploaded_zip = st.file_uploader("Upload Teachable Machine exported .zip", type=["zip"])

    if st.button("🚀 Run Evaluation & Post to Leaderboard", type="primary"):
        if not team_name:
            st.error("Please enter your Team Name.")
            return
        if not uploaded_zip:
            st.error("Please upload your exported model .zip file.")
            return

        with st.spinner("Unpacking model, loading locked test suite, and calculating ML metrics..."):
            try:
                zip_bytes = uploaded_zip.read()
                results = evaluator.evaluate_submission(zip_bytes, stage=stage)

                metrics = results["metrics"]
                errors = results["errors"]

                run_record = {
                    "stage": stage,
                    "team_name": team_name,
                    "model_version": model_version or "v1.0",
                    "notes": notes,
                    "metrics": metrics,
                    "errors": errors,
                }
                storage.save_run(run_record)

                st.success(f"🎉 Evaluation Complete! F1-Score: {metrics['f1']:.3f} | Accuracy: {metrics['accuracy'] * 100:.1f}%")

                col_res1, col_res2 = st.columns([1, 1])
                pos_label = "cat" if stage == "intro" else ("plant" if stage == "stage1" else "invasive")
                neg_label = "not_cat" if stage == "intro" else ("not_plant" if stage == "stage1" else "non_invasive")

                with col_res1:
                    render_confusion_matrix(metrics, pos_label=pos_label, neg_label=neg_label)
                with col_res2:
                    st.write("<b>Your Model's Bloopers (Errors to study for next time):</b>", unsafe_allow_html=True)
                    if not errors:
                        st.balloons()
                        st.success("No mistakes! 100% test accuracy!")
                    else:
                        for err in errors[:3]:
                            caption = f"{err.get('error_type')}: True='{err.get('true_label')}', Pred='{err.get('predicted_label')}' ({err.get('confidence') * 100:.1f}% conf)"
                            if err.get("image_path") and os.path.exists(err.get("image_path")):
                                st.image(err.get("image_path"), caption=caption, width=200)
                            else:
                                st.write(f"- {caption} ({err.get('filename')})")

            except Exception as e:
                st.error(f"Evaluation Error: {str(e)}")

def render_pipeline_board(storage: StorageManager, evaluator: ModelEvaluator):
    st.header("🏆 Board 3: Field Robot Pipeline Simulation")
    st.markdown(
        """
        **End-to-End Robotics Test:** Combines your Stage 1 Filter (*"Is it a plant?"*) with your 
        Stage 2 Classifier (*"Is it invasive?"*) to test how your Lego robot performs in the wild!
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("Pipeline Team Name", placeholder="e.g., Terratron Champions")
        stage1_file = st.file_uploader("Upload Stage 1 Model (.zip)", type=["zip"], key="p_s1")
    with col2:
        version_label = st.text_input("Pipeline Run Label", placeholder="e.g., Robot Run 1")
        stage2_file = st.file_uploader("Upload Stage 2 Model (.zip)", type=["zip"], key="p_s2")

    if st.button("🤖 Simulate Field Robot Test", type="primary"):
        if not stage1_file or not stage2_file:
            st.error("Please upload BOTH Stage 1 and Stage 2 models to test the pipeline.")
            return

        with st.spinner("Simulating robot traversal across field samples..."):
            try:
                pe = PipelineEvaluator(evaluator)
                pipeline_run = pe.evaluate_pipeline(stage1_file.read(), stage2_file.read())
                pipeline_metrics = pipeline_run["metrics"]

                storage.save_run({
                    "stage": "pipeline",
                    "team_name": team_name or "Terratron Robot",
                    "model_version": version_label or "Pipeline v1",
                    "notes": "Cascaded Field Robot Test",
                    "metrics": pipeline_metrics,
                    "errors": pipeline_run["errors"],
                })

                st.success(
                    f"🎉 Field Robot Test Complete! Combined F1: {pipeline_metrics['f1']:.3f} | "
                    f"Accuracy: {pipeline_metrics['accuracy'] * 100:.1f}% | "
                    f"Precision: {pipeline_metrics['precision']:.3f} | "
                    f"Recall: {pipeline_metrics['recall']:.3f}"
                )
                c1, c2 = st.columns(2)
                c1.metric("🌿 Stage 1 Accuracy", f"{pipeline_metrics['stage1_acc'] * 100:.1f}%")
                c2.metric("⚠️ Stage 2 Accuracy", f"{pipeline_metrics['stage2_acc'] * 100:.1f}%")

            except Exception as e:
                st.error(f"Pipeline Execution Error: {str(e)}")

    st.subheader("Field Robot Simulation Leaderboard")
    render_board(storage, stage="pipeline", stage_name="Field Robot", pos_label="invasive", neg_label="safe/harmless")

def render_admin_tab(dataset_mgr: DatasetManager):
    st.header("🛠️ Coach Dataset Administration")
    st.markdown(
        "Manage image intake, downsampling (to keep dataset < 25 MB), stratified 70/30 splitting, "
        "and generate student training packs."
    )

    pin = st.text_input("Enter Coach Access PIN", type="password")
    if pin != "fll2026" and pin != "coach":
        st.warning("Please enter the coach PIN (`fll2026`) to access dataset administration.")
        return

    st.success("🔓 Coach Access Granted")
    stats = dataset_mgr.get_dataset_stats()

    st.subheader("📊 Current Raw Dataset Pool (`data/raw/`)")
    col0, col1, col2 = st.columns(3)
    with col0:
        st.markdown("**🐱 Tutorial (Cat vs Not-Cat)**")
        st.write(f"- `cat/`: **{stats.get('intro', {}).get('cat', 0)}** photos")
        st.write(f"- `not_cat/`: **{stats.get('intro', {}).get('not_cat', 0)}** photos")
        if st.button("⚡ Fetch Cat vs. Not-Cat from Hugging Face"):
            with st.spinner("Fetching samples from Hugging Face Datasets (Bingsu/Cat_and_Dog)..."):
                seed_res = seed_cat_dataset()
                st.success(f"Fetched {seed_res['cat_saved']} cats and {seed_res['not_cat_saved']} not-cats from {seed_res['source']}!")
                st.rerun()

    with col1:
        st.markdown("**🌿 Stage 1 (Plant vs Not-Plant)**")
        st.write(f"- `plant/`: **{stats['stage1']['plant']}** photos")
        st.write(f"- `not_plant/`: **{stats['stage1']['not_plant']}** photos")
    with col2:
        st.markdown("**⚠️ Stage 2 (Invasive vs Non-Invasive)**")
        st.write(f"- `invasive/`: **{stats['stage2']['invasive']}** photos")
        st.write(f"- `non_invasive/`: **{stats['stage2']['non_invasive']}** photos")

    st.divider()
    st.subheader("⚙️ Generate 70% Train / 30% Locked Test Splits")
    st.write(
        "Clicking below will:\n"
        "1. Check all photos and downsample to $448 \\times 448$ (featherweight storage).\n"
        "2. Save 30% into locked `data/test_dataset/` for benchmarking.\n"
        "3. Package 70% into student training `.zip` bundles."
    )

    test_ratio = st.slider("Test Set Split Ratio", min_value=0.15, max_value=0.40, value=0.30, step=0.05)

    if st.button("⚡ Process & Generate Splits", type="primary"):
        with st.spinner("Processing, downsampling, and generating packs..."):
            summary = dataset_mgr.process_and_split_dataset(test_ratio=test_ratio)
            packs = dataset_mgr.create_student_training_packs()
            st.success("✅ Dataset split generated successfully!")
            st.json(summary)

    st.divider()
    st.subheader("📦 Download Student Training Packs")
    packs_dir = dataset_mgr.packs_dir
    p0 = os.path.join(packs_dir, "intro_train.zip")
    p1 = os.path.join(packs_dir, "stage1_train.zip")
    p2 = os.path.join(packs_dir, "stage2_train.zip")

    col_d0, col_d1, col_d2 = st.columns(3)
    with col_d0:
        if os.path.exists(p0):
            with open(p0, "rb") as f:
                st.download_button(
                    label="📥 Download Tutorial Pack (.zip)",
                    data=f.read(),
                    file_name="intro_train.zip",
                    mime="application/zip",
                )
        else:
            st.info("Tutorial training pack not generated yet.")

    with col_d1:
        if os.path.exists(p1):
            with open(p1, "rb") as f:
                st.download_button(
                    label="📥 Download Stage 1 Training Pack (.zip)",
                    data=f.read(),
                    file_name="stage1_train.zip",
                    mime="application/zip",
                )
        else:
            st.info("Stage 1 training pack not generated yet.")

    with col_d2:
        if os.path.exists(p2):
            with open(p2, "rb") as f:
                st.download_button(
                    label="📥 Download Stage 2 Training Pack (.zip)",
                    data=f.read(),
                    file_name="stage2_train.zip",
                    mime="application/zip",
                )
        else:
            st.info("Stage 2 training pack not generated yet.")

    st.divider()
    st.subheader("🤖 Coach P2: Gemini 2.5 Flash VLM Benchmark")
    st.write(
        "Run the state-of-the-art Gemini 2.5 Flash foundation model with botanical System Instructions "
        "across the locked test set to establish an industry reference bar on the Leaderboard."
    )
    api_key_input = st.text_input("Gemini API Key (optional if GEMINI_API_KEY is in environment)", type="password")
    bench_stage = st.selectbox("Stage to benchmark", options=["stage1", "stage2"])

    if st.button("⚡ Run Gemini Benchmark & Cache", type="secondary"):
        with st.spinner("Connecting to Gemini API and running zero-shot botanical classification..."):
            try:
                runner = GeminiBenchmarkRunner(api_key=api_key_input or None)
                bench_res = runner.evaluate_stage_benchmark(stage=bench_stage)
                st.success(f"✅ Gemini Benchmark Recorded! F1: {bench_res['metrics']['f1']:.3f} | Accuracy: {bench_res['metrics']['accuracy'] * 100:.1f}%")
                st.json(bench_res["metrics"])
            except Exception as e:
                st.error(f"Gemini Benchmark Error: {str(e)}")

def main():
    st.title("🌿 MIPS: Invasive Plant Species ML Platform")
    st.caption("First Lego League (FLL) Team 🤖 | Machine Learning Experiment Tracker & Leaderboards")

    storage = StorageManager()
    dataset_mgr = DatasetManager()
    evaluator = ModelEvaluator()

    tab_intro, tab_b1, tab_b2, tab_b3, tab_sub, tab_admin = st.tabs([
        "🐱 Tutorial: Cat vs Not-Cat",
        "🌿 Board 1: Plant vs Not-Plant",
        "⚠️ Board 2: Invasive vs Non-Invasive",
        "🏆 Board 3: Field Robot Pipeline",
        "📤 Submit Model",
        "🛠️ Coach Dataset Admin",
    ])

    with tab_intro:
        st.markdown(
            """
            > **Welcome to the MIPS Machine Learning Tutorial!** 🎓  
            > Practice the entire machine learning workflow using the **Cat vs. Not-Cat** dataset before moving to invasive plants.  
            > 
            > ⚠️ **Important Step for Google Teachable Machine:**  
            > Teachable Machine accepts **image files**, not `.zip` files!  
            > 1. **Download & UNZIP** `intro_train.zip` on your computer (double-click it to extract the `cat` and `not_cat` folders).  
            > 2. Open [Google Teachable Machine (Image Project)](https://teachablemachine.withgoogle.com/).  
            > 3. Name your classes `cat` and `not_cat`, then drag/upload the images from each folder into its matching class.  
            > 4. Click **Train Model**, then **Export Model**:  
            >    - **Option 1 (Recommended):** **TensorFlow** tab $\\rightarrow$ select **Keras** $\\rightarrow$ **Download my model**.  
            >    - **Option 2 (Fast):** **TensorFlow Lite** tab $\\rightarrow$ select **Floating point** $\\rightarrow$ **Download my model**.  
            >    - *(Do not pick TensorFlow.js, SavedModel, or EdgeTPU).*  
            > 5. Upload that downloaded `.zip` file in the **📤 Submit Model** tab to evaluate!
            """
        )
        p0 = os.path.join(dataset_mgr.packs_dir, "intro_train.zip")
        if os.path.exists(p0):
            with open(p0, "rb") as f:
                st.download_button(
                    label="📥 Download Tutorial Training Pack (intro_train.zip)",
                    data=f.read(),
                    file_name="intro_train.zip",
                    mime="application/zip",
                    key="dl_intro_tab",
                )
        render_board(storage, stage="intro", stage_name="🐱 Tutorial: Cat vs Not-Cat", pos_label="cat", neg_label="not_cat")

    with tab_b1:
        render_board(storage, stage="stage1", stage_name="🌿 Stage 1 (Plant vs Not-Plant)", pos_label="plant", neg_label="not_plant")

    with tab_b2:
        render_board(storage, stage="stage2", stage_name="⚠️ Stage 2 (Invasive vs Non-Invasive)", pos_label="invasive", neg_label="non_invasive")

    with tab_b3:
        render_pipeline_board(storage, evaluator)

    with tab_sub:
        render_submit_tab(storage, evaluator)

    with tab_admin:
        render_admin_tab(dataset_mgr)

if __name__ == "__main__":
    main()

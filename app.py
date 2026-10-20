import os
from datetime import datetime
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from core.dataset_manager import DatasetManager
from core.evaluator import ModelEvaluator
from core.storage import StorageManager

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

    if not leaderboard:
        st.info(f"No models submitted to {stage_name} yet. Be the first team to submit!")
        return

    # Top summary metrics
    top_run = leaderboard[0]
    top_metrics = top_run.get("metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🥇 Top F1-Score", f"{top_metrics.get('f1', 0.0):.3f}")
    col2.metric("🎯 Top Accuracy", f"{top_metrics.get('accuracy', 0.0) * 100:.1f}%")
    col3.metric("🏆 Top Team", f"{top_run.get('team_name', 'Unknown')} ({top_run.get('model_version', '')})")
    col4.metric("📊 Total Submissions", len(leaderboard))

    # Leaderboard table
    rows = []
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
            "id": run.get("id"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.drop(columns=["id"]),
        use_container_width=True,
        hide_index=True,
    )

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
    st.subheader(f"🔍 Inspecting Rank #1: {top_run.get('team_name')} ({top_run.get('model_version')})")
    col_cm, col_err = st.columns([1, 1])
    with col_cm:
        render_confusion_matrix(top_metrics, pos_label=pos_label, neg_label=neg_label)

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
    st.markdown(
        """
        1. Train your model in [Google Teachable Machine](https://teachablemachine.withgoogle.com/).
        2. Click **Export Model** -> **TensorFlow** -> **Keras** (or **TensorFlow Lite**) -> **Download my model**.
        3. Upload your downloaded `.zip` file below!
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("Team Name", placeholder="e.g., Team Kudzu Hunters")
        model_version = st.text_input("Model Version", placeholder="e.g., v1.2")
    with col2:
        stage = st.selectbox(
            "Which board are you submitting to?",
            options=["stage1", "stage2"],
            format_func=lambda x: "🌿 Stage 1: Plant vs Not-Plant" if x == "stage1" else "⚠️ Stage 2: Invasive vs Non-Invasive",
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
                pos_label = "plant" if stage == "stage1" else "invasive"
                neg_label = "not_plant" if stage == "stage1" else "non_invasive"

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

def main():
    st.title("🌿 MIPS: Invasive Plant Species ML Platform")
    st.caption("First Lego League (FLL) Team 🤖 | Machine Learning Experiment Tracker & Leaderboards")

    storage = StorageManager()
    dataset_mgr = DatasetManager()
    evaluator = ModelEvaluator()

    tab_b1, tab_b2, tab_b3, tab_sub, tab_admin = st.tabs([
        "🌿 Board 1: Plant vs Not-Plant",
        "⚠️ Board 2: Invasive vs Non-Invasive",
        "🏆 Board 3: Field Robot Pipeline",
        "📤 Submit Model",
        "🛠️ Coach Dataset Admin",
    ])

    with tab_b1:
        render_board(storage, stage="stage1", stage_name="🌿 Stage 1 (Plant vs Not-Plant)", pos_label="plant", neg_label="not_plant")

    with tab_b2:
        render_board(storage, stage="stage2", stage_name="⚠️ Stage 2 (Invasive vs Non-Invasive)", pos_label="invasive", neg_label="non_invasive")

    with tab_b3:
        st.header("🏆 Board 3: Field Robot Pipeline Simulation")
        st.info("Cascaded evaluation of Stage 1 filter followed by Stage 2 classifier. (See Task 5).")

    with tab_sub:
        render_submit_tab(storage, evaluator)

    with tab_admin:
        st.header("🛠️ Coach Dataset Administration")
        st.info("Manage image intake, 70/30 stratified splitting, and student training packs. (See Task 5).")

if __name__ == "__main__":
    main()

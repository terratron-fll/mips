import os
import textwrap
import uuid
from datetime import datetime
from typing import Optional
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st

from core.dataset_config import DatasetConfig, DatasetRegistry, StageConfig
from core.dataset_manager import DatasetManager
from core.evaluator import ModelEvaluator
from core.gemini_benchmark import GeminiBenchmarkRunner
from core.pipeline import PipelineEvaluator
from core.r2_sync import R2SyncManager
from core.storage import StorageManager, verify_admin_password
from scripts.seed_intro_data import seed_cat_dataset

st.set_page_config(
    page_title="MIPS: Machine Learning Leaderboard",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource
def get_r2_sync() -> R2SyncManager:
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}
    sync = R2SyncManager.from_secrets(secrets, local_data_dir="data")
    if sync.is_configured():
        sync.pull_metadata()
    return sync


def ensure_dataset_synced(r2_sync: R2SyncManager, dataset_id: str, force: bool = False) -> None:
    if not r2_sync.is_configured():
        return
    pulled = st.session_state.setdefault("pulled_datasets", set())
    if force or dataset_id not in pulled:
        r2_sync.pull_dataset_prefix(dataset_id, "test_dataset")
        r2_sync.pull_dataset_prefix(dataset_id, "gemini_cache")
        pulled.add(dataset_id)


def is_admin_unlocked() -> bool:
    return bool(st.session_state.get("gemini_admin_unlocked", False))


def render_admin_gate(key_prefix: str) -> bool:
    if is_admin_unlocked():
        col_a, col_b = st.columns([4, 1])
        col_a.success("🔓 Admin Unlocked (Live Gemini & Force Re-Split enabled)")
        if col_b.button("🔒 Lock", key=f"{key_prefix}_lock"):
            st.session_state["gemini_admin_unlocked"] = False
            st.rerun()
        return True

    with st.expander("🔐 Admin Unlock Required (Teacher / TA)", expanded=False):
        pwd = st.text_input("Admin Password", type="password", key=f"{key_prefix}_pwd")
        if st.button("Unlock Admin Controls", key=f"{key_prefix}_unlock"):
            try:
                expected = st.secrets.get("GEMINI_ADMIN_PASSWORD", "")
            except Exception:
                expected = ""
            if not expected:
                expected = os.environ.get("GEMINI_ADMIN_PASSWORD", "")
            if verify_admin_password(pwd, expected):
                st.session_state["gemini_admin_unlocked"] = True
                st.success("Admin unlocked!")
                st.rerun()
            else:
                st.error("Invalid password or GEMINI_ADMIN_PASSWORD not configured.")
    return False

def render_confusion_matrix(metrics: dict, pos_label: str, neg_label: str, key: str = "cm_chart"):
    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    tn = metrics.get("tn", 0)

    # Clean label formatting dynamically for any stage (e.g., cat, plant, invasive)
    pos_clean = pos_label.replace("_", " ").title()
    neg_clean = neg_label.replace("_", " ").title()

    # Dynamic status colors (green for correct, amber/red for mistakes)
    fp_color = "#f59e0b" if fp > 0 else "#9ca3af"
    fp_bg = "rgba(245, 158, 11, 0.12)" if fp > 0 else "rgba(156, 163, 175, 0.08)"
    
    fn_color = "#ef4444" if fn > 0 else "#9ca3af"
    fn_bg = "rgba(239, 68, 68, 0.12)" if fn > 0 else "rgba(156, 163, 175, 0.08)"

    tp_color = "#10b981"
    tp_bg = "rgba(16, 185, 129, 0.12)"

    tn_color = "#10b981"
    tn_bg = "rgba(16, 185, 129, 0.12)"

    card_html = textwrap.dedent(f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; margin: 8px 0 16px 0;">
        <div style="font-weight: 700; font-size: 15px; margin-bottom: 8px;">📊 2x2 Performance Scorecard</div>
        
        <!-- Column Headers (Predictions) -->
        <div style="display: grid; grid-template-columns: 110px 1fr 1fr; gap: 8px; margin-bottom: 6px; text-align: center; font-weight: 700; font-size: 12px; opacity: 0.85;">
            <div></div>
            <div style="background: rgba(128, 128, 128, 0.12); padding: 6px; border-radius: 6px;">PREDICTED:<br>{neg_clean}</div>
            <div style="background: rgba(128, 128, 128, 0.12); padding: 6px; border-radius: 6px;">PREDICTED:<br>{pos_clean}</div>
        </div>

        <!-- Row 1: Actual Negative -->
        <div style="display: grid; grid-template-columns: 110px 1fr 1fr; gap: 8px; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; justify-content: center; background: rgba(128, 128, 128, 0.12); border-radius: 6px; font-weight: 700; font-size: 12px; text-align: center; padding: 6px; opacity: 0.85;">
                ACTUAL:<br>{neg_clean}
            </div>
            <!-- TN Card -->
            <div style="background: {tn_bg}; border: 2px solid {tn_color}; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; font-weight: 800; color: {tn_color}; letter-spacing: 0.5px;">✅ TRUE NEGATIVE</div>
                <div style="font-size: 32px; font-weight: 900; line-height: 1.1; margin: 4px 0;">{tn}</div>
                <div style="font-size: 11px; opacity: 0.85;">Correctly called {neg_clean}</div>
            </div>
            <!-- FP Card -->
            <div style="background: {fp_bg}; border: 2px solid {fp_color}; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; font-weight: 800; color: {fp_color}; letter-spacing: 0.5px;">⚠️ FALSE ALARM (FP)</div>
                <div style="font-size: 32px; font-weight: 900; line-height: 1.1; margin: 4px 0;">{fp}</div>
                <div style="font-size: 11px; opacity: 0.85;">Mistakenly called {pos_clean}</div>
            </div>
        </div>

        <!-- Row 2: Actual Positive -->
        <div style="display: grid; grid-template-columns: 110px 1fr 1fr; gap: 8px;">
            <div style="display: flex; align-items: center; justify-content: center; background: rgba(128, 128, 128, 0.12); border-radius: 6px; font-weight: 700; font-size: 12px; text-align: center; padding: 6px; opacity: 0.85;">
                ACTUAL:<br>{pos_clean}
            </div>
            <!-- FN Card -->
            <div style="background: {fn_bg}; border: 2px solid {fn_color}; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; font-weight: 800; color: {fn_color}; letter-spacing: 0.5px;">❌ MISSED TARGET (FN)</div>
                <div style="font-size: 32px; font-weight: 900; line-height: 1.1; margin: 4px 0;">{fn}</div>
                <div style="font-size: 11px; opacity: 0.85;">Actually {pos_clean}, but missed</div>
            </div>
            <!-- TP Card -->
            <div style="background: {tp_bg}; border: 2px solid {tp_color}; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; font-weight: 800; color: {tp_color}; letter-spacing: 0.5px;">🎯 TRUE POSITIVE</div>
                <div style="font-size: 32px; font-weight: 900; line-height: 1.1; margin: 4px 0;">{tp}</div>
                <div style="font-size: 11px; opacity: 0.85;">Correctly caught {pos_clean}</div>
            </div>
        </div>
    </div>
    """).strip()

    if hasattr(st, "html"):
        st.html(card_html)
    else:
        st.markdown(card_html, unsafe_allow_html=True)

def render_board(
    storage: StorageManager,
    dataset_config: DatasetConfig,
    stage: str,
    stage_name: str,
    pos_label: str,
    neg_label: str,
    r2_sync: Optional[R2SyncManager] = None,
):
    st.header(f"{stage_name} Leaderboard")
    current_split = dataset_config.split_version
    split_choice = st.radio(
        "Test Split Version",
        options=["current", "all"],
        format_func=lambda x: f"Current Split (v{current_split})" if x == "current" else "All Split Versions",
        horizontal=True,
        key=f"split_filter_{dataset_config.id}_{stage}",
    )
    split_filter = current_split if split_choice == "current" else None
    leaderboard = storage.get_leaderboard(
        stage=stage,
        dataset_id=dataset_config.id,
        split_version=split_filter,
    )

    # Check for cached Gemini benchmarks
    gemini_runner = GeminiBenchmarkRunner(dataset_config=dataset_config, r2_sync=r2_sync)
    gemini_entries = gemini_runner.get_stage_benchmarks(stage)

    if not leaderboard and not gemini_entries:
        st.info(f"No models submitted to {stage_name} (split v{current_split}) yet. Be the first team to submit!")
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
    for gemini_entry in gemini_entries:
        gm = gemini_entry.get("metrics", {})
        gemini_label = gemini_entry.get("model_name", "Gemini 3.5 Flash-Lite")
        gemini_version = gemini_entry.get("version", "Zero-Shot VLM (SI)")
        rows.append({
            "Rank": "🤖 BENCHMARK",
            "Team": gemini_label,
            "Version": gemini_version,
            "Split": f"v{gemini_entry.get('split_version', current_split)}",
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
            "Split": f"v{run.get('split_version', 1)}",
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

    # Build ordered list of inspectable run objects matching the table rows
    inspectable_runs = []
    for gemini_entry in gemini_entries:
        gemini_label = gemini_entry.get("model_name", "Gemini 3.5 Flash-Lite")
        gemini_version = gemini_entry.get("version", "Zero-Shot VLM")
        inspectable_runs.append({
            "label": f"🤖 Benchmark: {gemini_label} ({gemini_version})",
            "run": gemini_entry,
        })

    for rank, run in enumerate(leaderboard, 1):
        team = run.get("team_name", "Team")
        ver = run.get("model_version", "")
        f1 = run.get("metrics", {}).get("f1", 0.0)
        inspectable_runs.append({
            "label": f"#{rank}: {team} ({ver}) — F1: {f1:.3f}",
            "run": run,
        })

    st.caption("💡 **Tip:** Click on any row in the table below to inspect its Confusion Matrix & Bloopers!")
    selection = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"leaderboard_table_{stage}",
    )

    # Determine which run is selected
    selected_idx = 0
    if selection:
        sel = getattr(selection, "selection", selection)
        if isinstance(sel, dict):
            selected_rows = sel.get("rows", [])
        else:
            selected_rows = getattr(sel, "rows", [])
        if selected_rows and len(selected_rows) > 0:
            row_clicked = selected_rows[0]
            if 0 <= row_clicked < len(inspectable_runs):
                selected_idx = row_clicked
                st.session_state[f"selected_run_idx_{stage}"] = selected_idx
        elif f"selected_run_idx_{stage}" in st.session_state:
            saved_idx = st.session_state[f"selected_run_idx_{stage}"]
            if 0 <= saved_idx < len(inspectable_runs):
                selected_idx = saved_idx

    # Hill-climbing line chart
    st.subheader("📈 Hill-Climbing Progress Over Time")
    history = storage.get_history(
        stage=stage,
        dataset_id=dataset_config.id,
        split_version=split_filter,
    )
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
        st.plotly_chart(fig_line, use_container_width=True, key=f"progress_chart_{stage}")

    # Detailed inspection of selected run
    if inspectable_runs:
        st.divider()
        st.subheader("🔍 Run Deep Dive & Bloopers Inspection")

        chosen_idx = st.selectbox(
            "👉 Select a submission to inspect its Confusion Matrix & Bloopers:",
            options=list(range(len(inspectable_runs))),
            format_func=lambda i: inspectable_runs[i]["label"],
            index=selected_idx,
            key=f"select_run_sb_{stage}",
        )
        if chosen_idx != selected_idx:
            st.session_state[f"selected_run_idx_{stage}"] = chosen_idx
            selected_idx = chosen_idx

        target_item = inspectable_runs[selected_idx]
        target_run = target_item["run"]
        run_id = target_run.get("id")

        notes = target_run.get("notes")
        if notes:
            st.info(f"📝 **Team Hypothesis & Experiment Notes:** {notes}")

        if target_run.get("prompt") or target_run.get("preset") or target_item["label"].startswith("🤖 Benchmark:"):
            preset_key = target_run.get("preset", "flash_lite")
            api_type = GeminiBenchmarkRunner.BENCHMARK_PRESETS.get(preset_key, {}).get("api_type", "generate_content")
            prompt_text = target_run.get("prompt") or gemini_runner.get_default_prompt(stage, api_type=api_type)
            if prompt_text:
                with st.expander("📝 Gemini Prompt / Instructions Used", expanded=False):
                    st.code(prompt_text, language="text")

        if stage == "pipeline":
            s1_art = storage.get_model_artifact_bytes(run_id, subdir="stage1")
            s2_art = storage.get_model_artifact_bytes(run_id, subdir="stage2")
            if s1_art or s2_art:
                dl_c1, dl_c2 = st.columns(2)
                if s1_art:
                    dl_c1.download_button(
                        "⬇️ Download Stage 1 Model (.zip)",
                        data=s1_art[1],
                        file_name=s1_art[0],
                        mime="application/zip",
                        key=f"dl_{stage}_{run_id}_s1_{selected_idx}",
                    )
                if s2_art:
                    dl_c2.download_button(
                        "⬇️ Download Stage 2 Model (.zip)",
                        data=s2_art[1],
                        file_name=s2_art[0],
                        mime="application/zip",
                        key=f"dl_{stage}_{run_id}_s2_{selected_idx}",
                    )
        else:
            artifact = storage.get_model_artifact_bytes(run_id)
            if artifact:
                st.download_button(
                    "⬇️ Download Saved Model (.zip)",
                    data=artifact[1],
                    file_name=artifact[0],
                    mime="application/zip",
                    key=f"dl_{stage}_{run_id}_{selected_idx}",
                )

        col_cm, col_err = st.columns([1, 1])
        with col_cm:
            render_confusion_matrix(
                target_run.get("metrics", {}),
                pos_label=pos_label,
                neg_label=neg_label,
                key=f"board_cm_{stage}_{selected_idx}",
            )

        with col_err:
            st.write("<b>Bloopers Gallery (Model Errors on Test Dataset):</b>", unsafe_allow_html=True)
            errors = target_run.get("errors", [])
            if not errors:
                st.success("🎉 Perfect score! No errors found on the test dataset.")
            else:
                st.caption(f"Showing up to 6 mistakes ({len(errors)} total errors):")
                err_cols = st.columns(min(3, len(errors)))
                for idx, err in enumerate(errors[:6]):
                    ecol = err_cols[idx % len(err_cols)]
                    with ecol:
                        err_path = err.get("image_path")
                        conf = err.get("confidence")
                        conf_str = f" ({conf * 100:.1f}%)" if conf is not None else ""
                        caption = f"{err.get('error_type')}: True='{err.get('true_label')}', Pred='{err.get('predicted_label')}'{conf_str}"
                        if err_path and os.path.exists(err_path):
                            st.image(err_path, caption=caption, use_container_width=True)
                        else:
                            st.write(f"- {caption} ({err.get('filename')})")

def render_submit_tab(
    storage: StorageManager,
    evaluator: ModelEvaluator,
    dataset_config: DatasetConfig,
    dataset_mgr: DatasetManager,
    r2_sync: Optional[R2SyncManager] = None,
):
    st.header(f"📤 Submit Your Teachable Machine Model ({dataset_config.icon} {dataset_config.name})")

    with st.expander("❓ Exactly Which Export Option to Pick in Google Teachable Machine", expanded=True):
        st.markdown(
            """
            ### 🌟 Recommended Export: TensorFlow Lite (Floating point)
            Google Teachable Machine's legacy Keras (`.h5`) format has known graph incompatibilities with modern TensorFlow.  
            **Always use TensorFlow Lite:**

            1. In Teachable Machine, click **Export Model**.
            2. Click the **TensorFlow Lite** tab at the top.
            3. Select the **Floating point** radio button.
            4. Click **Download my model** (this will download a `.zip` containing `model_unquant.tflite` and `labels.txt`).
            5. Drop that downloaded `.zip` file below!

            ---
            *Why TensorFlow Lite?* It is a self-contained FlatBuffer model that runs 5x faster, uses less memory, and completely avoids Keras Python version incompatibilities.
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("Team Name", placeholder="e.g., Team Kudzu Hunters")
        model_version = st.text_input("Model Version", placeholder="e.g., v1.2")
    stage_keys = list(dataset_config.stages.keys())
    with col2:
        stage = st.selectbox(
            "Which board are you submitting to?",
            options=stage_keys,
            format_func=lambda x: dataset_config.stages[x].name,
            key=f"submit_stage_{dataset_config.id}",
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
                if r2_sync is not None and r2_sync.is_configured():
                    ensure_dataset_synced(r2_sync, dataset_config.id)
                zip_bytes = uploaded_zip.read()
                run_id = f"run-{uuid.uuid4().hex[:8]}"
                save_dir = os.path.join(storage.models_dir, run_id)
                results = evaluator.evaluate_submission(zip_bytes, stage=stage, save_dir=save_dir)
                storage.save_model_artifact(run_id, zip_bytes)

                metrics = results["metrics"]
                errors = results["errors"]

                run_record = {
                    "id": run_id,
                    "dataset_id": dataset_config.id,
                    "split_version": dataset_config.split_version,
                    "stage": stage,
                    "team_name": team_name,
                    "model_version": model_version or "v1.0",
                    "notes": notes,
                    "metrics": metrics,
                    "errors": errors,
                    "artifacts": results.get("saved_artifacts", {}),
                }
                storage.save_run(run_record)

                st.success(f"🎉 Evaluation Complete! F1-Score: {metrics['f1']:.3f} | Accuracy: {metrics['accuracy'] * 100:.1f}%")

                col_res1, col_res2 = st.columns([1, 1])
                sc = dataset_config.stages[stage]
                pos_label = sc.pos_label
                neg_label = sc.neg_label

                with col_res1:
                    render_confusion_matrix(
                        metrics,
                        pos_label=pos_label,
                        neg_label=neg_label,
                        key=f"submit_cm_{stage}",
                    )
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

def render_pipeline_board(
    storage: StorageManager,
    evaluator: ModelEvaluator,
    dataset_config: DatasetConfig,
    r2_sync: Optional[R2SyncManager] = None,
):
    st.header("🏆 Field Robot Pipeline Simulation")
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
                if r2_sync is not None and r2_sync.is_configured():
                    ensure_dataset_synced(r2_sync, dataset_config.id)
                pe = PipelineEvaluator(evaluator)
                run_id = f"run-{uuid.uuid4().hex[:8]}"
                s1_bytes = stage1_file.read()
                s2_bytes = stage2_file.read()
                s1_save_dir = os.path.join(storage.models_dir, run_id, "stage1")
                s2_save_dir = os.path.join(storage.models_dir, run_id, "stage2")
                pipeline_run = pe.evaluate_pipeline(
                    s1_bytes,
                    s2_bytes,
                    stage1_save_dir=s1_save_dir,
                    stage2_save_dir=s2_save_dir,
                )
                storage.save_model_artifact(run_id, s1_bytes, subdir="stage1")
                storage.save_model_artifact(run_id, s2_bytes, subdir="stage2")
                pipeline_metrics = pipeline_run["metrics"]

                storage.save_run({
                    "id": run_id,
                    "dataset_id": dataset_config.id,
                    "split_version": dataset_config.split_version,
                    "stage": "pipeline",
                    "team_name": team_name or "Terratron Robot",
                    "model_version": version_label or "Pipeline v1",
                    "notes": "Cascaded Field Robot Test",
                    "metrics": pipeline_metrics,
                    "errors": pipeline_run["errors"],
                    "artifacts": {
                        "stage1": pipeline_run["stage1_result"].get("saved_artifacts", {}),
                        "stage2": pipeline_run["stage2_result"].get("saved_artifacts", {}),
                    },
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
    s2_stage = dataset_config.stages.get("stage2")
    pos_l = s2_stage.pos_label if s2_stage else "invasive"
    neg_l = f"not_{s2_stage.pos_label}" if s2_stage else "safe/harmless"
    render_board(
        storage,
        dataset_config=dataset_config,
        stage="pipeline",
        stage_name="Field Robot",
        pos_label=pos_l,
        neg_label=neg_l,
        r2_sync=r2_sync,
    )

STAGE_LABELS = {
    "intro": "🐱 Tutorial: Cat vs Not-Cat",
    "stage1": "🌿 Stage 1: Plant vs Not-Plant",
    "stage2": "⚠️ Stage 2: Invasive vs Non-Invasive",
}


def render_training_pack_download(
    dataset_mgr: DatasetManager,
    stage: str,
    label: str,
    key: str,
    empty_message: Optional[str] = None,
):
    pack_path = os.path.join(dataset_mgr.packs_dir, f"{stage}_train.zip")
    if not os.path.exists(pack_path) and dataset_mgr.r2_sync is not None and dataset_mgr.r2_sync.is_configured():
        dataset_mgr.r2_sync.ensure_training_pack_local(dataset_mgr.dataset_config.id, stage)
    if os.path.exists(pack_path):
        with open(pack_path, "rb") as f:
            st.download_button(
                label=label,
                data=f.read(),
                file_name=f"{stage}_train.zip",
                mime="application/zip",
                key=key,
            )
    else:
        st.info(
            empty_message
            or f"`{stage}_train.zip` has not been generated yet. Head to **🗂️ Dataset Manager** to upload photos and generate splits."
        )


def render_dataset_tab(
    dataset_mgr: DatasetManager,
    dataset_config: DatasetConfig,
    registry: DatasetRegistry,
):
    st.header(f"🗂️ Dataset Manager — {dataset_config.icon} {dataset_config.name}")
    st.caption(f"Dataset ID: `{dataset_config.id}` | Current Split Version: **v{dataset_config.split_version}**")
    st.markdown(
        "Upload new training photos, inspect dataset counts, generate incremental 70/30 train/test splits, "
        "and download student training packs."
    )

    stage_keys = list(dataset_config.stages.keys())
    st.subheader("📸 Upload Photos to Dataset")
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        upload_stage = st.selectbox(
            "Dataset Stage",
            options=stage_keys,
            format_func=lambda s: dataset_config.stages[s].name,
            key=f"dataset_upload_stage_{dataset_config.id}",
        )
    with up_col2:
        upload_class = st.selectbox(
            "Class Label",
            options=dataset_config.stages[upload_stage].classes,
            format_func=lambda c: c.replace("_", " ").title(),
            key=f"dataset_upload_class_{dataset_config.id}",
        )

    uploaded_photos = st.file_uploader(
        "Select photos to add (.jpg, .jpeg, .png, .webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"dataset_photo_uploader_{dataset_config.id}",
    )

    if st.button("📤 Add Photos to Dataset", type="primary", key=f"add_photos_btn_{dataset_config.id}"):
        if not uploaded_photos:
            st.warning("Please select at least one photo to upload.")
        else:
            file_payload = [(f.name, f.read()) for f in uploaded_photos]
            saved_count, upload_errors = dataset_mgr.save_uploaded_images(
                upload_stage, upload_class, file_payload
            )
            if saved_count:
                st.success(
                    f"✅ Added {saved_count} photo(s) to `{upload_stage}/{upload_class}`! "
                    "Click **⚡ Process Incremental Split** below to include them in the training packs and test set."
                )
            if upload_errors:
                st.error("Skipped invalid file(s): " + "; ".join(upload_errors))

    st.divider()
    stats = dataset_mgr.get_dataset_stats()

    st.subheader(f"📊 Current Raw Dataset Pool (`data/datasets/{dataset_config.id}/raw/`)")
    cols = st.columns(max(1, len(stage_keys)))
    for idx, skey in enumerate(stage_keys):
        sc = dataset_config.stages[skey]
        with cols[idx]:
            st.markdown(f"**{sc.name}**")
            st.write(f"- `{sc.pos_label}/`: **{stats.get(skey, {}).get(sc.pos_label, 0)}** photos")
            st.write(f"- `{sc.neg_label}/`: **{stats.get(skey, {}).get(sc.neg_label, 0)}** photos")
            if dataset_config.id == "tutorial_cat" and skey == "intro":
                if st.button("⚡ Fetch Cat vs. Not-Cat from Hugging Face", key="seed_hf_cat_btn"):
                    with st.spinner("Fetching samples from Hugging Face Datasets (Bingsu/Cat_and_Dog)..."):
                        intro_raw = os.path.join(dataset_mgr.raw_dir, "intro")
                        seed_res = seed_cat_dataset(target_dir=intro_raw)
                        if dataset_mgr.r2_sync is not None and dataset_mgr.r2_sync.is_configured():
                            for cls_name in sc.classes:
                                cls_dir = os.path.join(intro_raw, cls_name)
                                if os.path.isdir(cls_dir):
                                    for fname in os.listdir(cls_dir):
                                        fpath = os.path.join(cls_dir, fname)
                                        if os.path.isfile(fpath):
                                            dataset_mgr.r2_sync.push_file(
                                                f"datasets/{dataset_config.id}/raw/intro/{cls_name}/{fname}",
                                                fpath,
                                            )
                        st.success(f"Fetched {seed_res['cat_saved']} cats and {seed_res['not_cat_saved']} not-cats from {seed_res['source']}!")
                        st.rerun()

    st.divider()
    st.subheader("⚙️ Generate 70% Train / 30% Locked Test Splits")
    st.write(
        "Incremental split preserves all existing train/test assignments so leaderboard scores remain comparable, "
        "while appending newly uploaded photos (downsampled to $448 \\times 448$) and rebuilding student `.zip` packs."
    )

    test_ratio = st.slider(
        "Test Set Split Ratio",
        min_value=0.15,
        max_value=0.40,
        value=0.30,
        step=0.05,
        key=f"test_ratio_{dataset_config.id}",
    )

    if st.button("⚡ Process Incremental Split (Append New Photos)", type="primary", key=f"inc_split_{dataset_config.id}"):
        with st.spinner("Processing new photos and updating packs..."):
            summary = dataset_mgr.process_and_split_dataset(test_ratio=test_ratio, force_resplit=False)
            dataset_mgr.create_student_training_packs()
            st.success(f"✅ Incremental split complete (Split v{dataset_mgr.get_split_version()})!")
            st.json(summary)

    with st.expander("⚠️ Admin Only: Force Full Re-Split (New Split Version)", expanded=False):
        st.caption(
            "Re-shuffles all raw images into a brand-new train/test split and increments the dataset's `split_version`. "
            "Existing leaderboard runs stay tagged with their previous split version."
        )
        if render_admin_gate(f"resplit_{dataset_config.id}"):
            if st.button("🔄 Force Full Re-Split", type="secondary", key=f"force_resplit_{dataset_config.id}"):
                with st.spinner("Re-splitting entire dataset and bumping split version..."):
                    summary = dataset_mgr.process_and_split_dataset(test_ratio=test_ratio, force_resplit=True)
                    dataset_config.split_version = dataset_mgr.get_split_version()
                    registry.register_dataset(dataset_config)
                    dataset_mgr.create_student_training_packs()
                    st.success(f"✅ Full re-split complete! Now on Split v{dataset_config.split_version}.")
                    st.json(summary)

    st.divider()
    st.subheader("📦 Download Student Training Packs")
    pack_cols = st.columns(max(1, len(stage_keys)))
    for idx, skey in enumerate(stage_keys):
        sc = dataset_config.stages[skey]
        with pack_cols[idx]:
            render_training_pack_download(
                dataset_mgr,
                stage=skey,
                label=f"📥 Download {sc.name} Pack (.zip)",
                key=f"dl_admin_{dataset_config.id}_{skey}",
                empty_message=f"{sc.name} training pack not generated yet.",
            )

    st.divider()
    with st.expander("🤖 Coach Reference: Gemini Foundation Model Benchmarks", expanded=False):
        st.write(
            "Run **Gemini 3.5 Flash-Lite**, **Gemini 3.8 Flash**, and/or **Gemini Decision 1 (Decisions API)** "
            "across the locked test set to establish reference bars on the Leaderboard. "
            "Results are cached per dataset/stage and synced to R2 when configured."
        )
        if render_admin_gate(f"gemini_{dataset_config.id}"):
            api_key_input = st.text_input(
                "Gemini API Key (optional if set in .streamlit/secrets.toml or GEMINI_API_KEY)",
                type="password",
                key=f"gemini_key_{dataset_config.id}",
            )
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                bench_stage = st.selectbox(
                    "Stage to benchmark",
                    options=stage_keys,
                    format_func=lambda s: dataset_config.stages[s].name,
                    key=f"bench_stage_{dataset_config.id}",
                )
            with b_col2:
                preset_options = ["all"] + list(GeminiBenchmarkRunner.BENCHMARK_PRESETS.keys())
                bench_preset = st.selectbox(
                    "Gemini Benchmark Model / API",
                    options=preset_options,
                    format_func=lambda p: (
                        "All 3 Benchmarks (Flash-Lite + Flash + Decision 1)"
                        if p == "all"
                        else f"{GeminiBenchmarkRunner.BENCHMARK_PRESETS[p]['label']} ({GeminiBenchmarkRunner.BENCHMARK_PRESETS[p]['model']})"
                    ),
                    key=f"bench_preset_{dataset_config.id}",
                )

            preview_runner = GeminiBenchmarkRunner(
                api_key=api_key_input or None,
                dataset_config=dataset_config,
                r2_sync=dataset_mgr.r2_sync,
            )
            custom_prompt: Optional[str] = None
            if bench_preset == "all":
                st.markdown("**📝 Prompts / Instructions Sent to Gemini:**")
                st.caption("VLM System Instruction (used by Flash-Lite & Flash):")
                st.code(preview_runner.get_default_prompt(bench_stage, api_type="generate_content"), language="text")
                st.caption("Decisions API Instructions & Criteria (used by Decision 1):")
                st.code(preview_runner.get_default_prompt(bench_stage, api_type="decisions"), language="text")
            else:
                api_type = GeminiBenchmarkRunner.BENCHMARK_PRESETS[bench_preset]["api_type"]
                default_prompt = preview_runner.get_default_prompt(bench_stage, api_type=api_type)
                prompt_label = (
                    "📝 Decisions API Instructions (editable)"
                    if api_type == "decisions"
                    else "📝 VLM System Instruction Prompt (editable)"
                )
                if api_type == "decisions":
                    spec = preview_runner.get_decision_spec(bench_stage)
                    default_prompt = spec.get("instructions", "")
                    st.caption("Decisions API Criteria:")
                    st.json(spec.get("criteria", {}))
                custom_prompt = st.text_area(
                    prompt_label,
                    value=default_prompt,
                    height=110,
                    key=f"gemini_prompt_{dataset_config.id}_{bench_stage}_{bench_preset}",
                )

            if st.button("⚡ Run Gemini Benchmark & Cache", type="secondary", key=f"run_gemini_{dataset_config.id}"):
                targets = list(GeminiBenchmarkRunner.BENCHMARK_PRESETS.keys()) if bench_preset == "all" else [bench_preset]
                with st.spinner("Connecting to Gemini API and running benchmark evaluation..."):
                    try:
                        runner = GeminiBenchmarkRunner(
                            api_key=api_key_input or None,
                            dataset_config=dataset_config,
                            r2_sync=dataset_mgr.r2_sync,
                        )
                        for target_preset in targets:
                            bench_res = runner.evaluate_stage_benchmark(
                                stage=bench_stage,
                                test_images_dir=dataset_mgr.test_dir,
                                preset=target_preset,
                                custom_prompt=custom_prompt,
                            )
                            st.success(
                                f"✅ {bench_res['model_name']} Recorded! "
                                f"F1: {bench_res['metrics']['f1']:.3f} | Accuracy: {bench_res['metrics']['accuracy'] * 100:.1f}%"
                            )
                            st.json(bench_res["metrics"])
                    except Exception as e:
                        st.error(f"Gemini Benchmark Error: {str(e)}")


def render_sidebar(r2_sync: R2SyncManager, registry: DatasetRegistry) -> DatasetConfig:
    st.sidebar.header("📂 Datasets")
    datasets = registry.list_datasets()
    ds_map = {d.id: d for d in datasets}
    ds_ids = [d.id for d in datasets]

    default_id = st.session_state.get("active_dataset_id", ds_ids[0])
    if default_id not in ds_map:
        default_id = ds_ids[0]

    selected_id = st.sidebar.selectbox(
        "Active Dataset",
        options=ds_ids,
        index=ds_ids.index(default_id),
        format_func=lambda did: f"{ds_map[did].icon} {ds_map[did].name}",
        key="sidebar_dataset_selector",
    )
    st.session_state["active_dataset_id"] = selected_id
    active_dataset = ds_map[selected_id]

    with st.sidebar.expander("➕ Create New Dataset", expanded=False):
        with st.form("create_dataset_form"):
            new_name = st.text_input("Dataset Name", placeholder="e.g., Waste Sorter")
            new_slug_raw = st.text_input("Dataset Slug (optional)", placeholder="auto-generated if blank")
            new_icon = st.text_input("Icon Emoji", value="📦")
            new_desc = st.text_input("Description", placeholder="Short description of the challenge")
            num_stages = st.radio("Number of Stages", options=[1, 2], horizontal=True)

            st.markdown("**Stage 1**")
            s1_name = st.text_input("Stage 1 Name", value="Stage 1: Primary Classifier")
            s1_pos = st.text_input("Stage 1 Positive Label", value="target")
            s1_neg = st.text_input("Stage 1 Negative Label", value="other")

            s2_name = st.text_input("Stage 2 Name (if 2 stages)", value="Stage 2: Fine Classifier")
            s2_pos = st.text_input("Stage 2 Positive Label", value="positive")
            s2_neg = st.text_input("Stage 2 Negative Label", value="negative")
            enable_pipeline = st.checkbox("Enable Cascaded Pipeline Board (2-stage only)", value=False)

            submitted = st.form_submit_button("Create Dataset")
            if submitted:
                try:
                    slug = (new_slug_raw.strip() or DatasetRegistry.slugify(new_name))
                    DatasetRegistry.validate_slug(slug)
                    if slug in ds_map:
                        st.error(f"Dataset '{slug}' already exists.")
                    elif not new_name.strip():
                        st.error("Dataset Name is required.")
                    else:
                        stages = {
                            "stage1": StageConfig(
                                stage_id="stage1",
                                name=s1_name.strip() or "Stage 1",
                                pos_label=DatasetRegistry.slugify(s1_pos),
                                neg_label=DatasetRegistry.slugify(s1_neg),
                            )
                        }
                        if num_stages == 2:
                            stages["stage2"] = StageConfig(
                                stage_id="stage2",
                                name=s2_name.strip() or "Stage 2",
                                pos_label=DatasetRegistry.slugify(s2_pos),
                                neg_label=DatasetRegistry.slugify(s2_neg),
                            )
                        created = DatasetConfig(
                            id=slug,
                            name=new_name.strip(),
                            icon=new_icon.strip() or "📦",
                            description=new_desc.strip(),
                            stages=stages,
                            has_pipeline=(num_stages == 2 and enable_pipeline),
                            split_version=1,
                        )
                        registry.register_dataset(created)
                        st.session_state["active_dataset_id"] = created.id
                        st.success(f"Created dataset '{created.name}'!")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not create dataset: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("☁️ Storage & Sync")
    if r2_sync.is_configured():
        st.sidebar.success(f"R2 Connected (`{r2_sync.bucket}`)")
        if r2_sync.last_sync_at:
            st.sidebar.caption(f"Last sync: {r2_sync.last_sync_at[:19].replace('T', ' ')} ({r2_sync.last_sync_status})")
        if r2_sync.last_sync_error:
            st.sidebar.warning(f"Sync warning: {r2_sync.last_sync_error}")
        if st.sidebar.button("🔄 Sync Now", key="sidebar_sync_now"):
            r2_sync.pull_metadata()
            ensure_dataset_synced(r2_sync, active_dataset.id, force=True)
            st.rerun()
    else:
        st.sidebar.info("Local Storage Mode (R2 not configured)")

    return active_dataset


def main():
    r2_sync = get_r2_sync()
    registry = DatasetRegistry(datasets_dir="data/datasets", r2_sync=r2_sync)
    active_dataset = render_sidebar(r2_sync, registry)
    ensure_dataset_synced(r2_sync, active_dataset.id)

    st.title(f"{active_dataset.icon} MIPS: {active_dataset.name}")
    subtitle = active_dataset.description or "First Lego League (FLL) Team 🤖 | Machine Learning Experiment Tracker & Leaderboards"
    st.caption(f"{subtitle} | Split v{active_dataset.split_version}")

    storage = StorageManager(r2_sync=r2_sync)
    dataset_mgr = DatasetManager(dataset_config=active_dataset, r2_sync=r2_sync)
    evaluator = ModelEvaluator(test_dataset_dir=dataset_mgr.test_dir, dataset_config=active_dataset)

    stage_items = list(active_dataset.stages.items())
    tab_labels = [sc.name for _, sc in stage_items]
    if active_dataset.has_pipeline:
        tab_labels.append("🏆 Field Robot Pipeline")
    tab_labels.extend(["📤 Submit Model", "🗂️ Dataset Manager"])

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    for skey, sc in stage_items:
        with tabs[tab_idx]:
            if active_dataset.id == "tutorial_cat" and skey == "intro":
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
                    >    - Choose **TensorFlow Lite** tab $\\rightarrow$ select **Floating point** $\\rightarrow$ **Download my model**.  
                    >    - *(Avoid Keras, TensorFlow.js, SavedModel, or EdgeTPU).*  
                    > 5. Upload that downloaded `.zip` file in the **📤 Submit Model** tab to evaluate!
                    """
                )
            render_training_pack_download(
                dataset_mgr,
                stage=skey,
                label=f"📥 Download {sc.name} Training Pack ({skey}_train.zip)",
                key=f"dl_{active_dataset.id}_{skey}_tab",
            )
            render_board(
                storage,
                dataset_config=active_dataset,
                stage=skey,
                stage_name=sc.name,
                pos_label=sc.pos_label,
                neg_label=sc.neg_label,
                r2_sync=r2_sync,
            )
        tab_idx += 1

    if active_dataset.has_pipeline:
        with tabs[tab_idx]:
            render_pipeline_board(storage, evaluator, dataset_config=active_dataset, r2_sync=r2_sync)
        tab_idx += 1

    with tabs[tab_idx]:
        render_submit_tab(storage, evaluator, dataset_config=active_dataset, dataset_mgr=dataset_mgr, r2_sync=r2_sync)
    tab_idx += 1

    with tabs[tab_idx]:
        render_dataset_tab(dataset_mgr, dataset_config=active_dataset, registry=registry)


if __name__ == "__main__":
    main()

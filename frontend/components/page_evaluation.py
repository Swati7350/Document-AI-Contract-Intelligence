import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from processors.evaluation import compute_metrics, TEST_DATASET


def render() -> None:
    st.markdown('<div class="sec-header"><div class="sec-title">📊 Evaluation Dashboard</div><div class="sec-sub">Field-level extraction accuracy across a verified test dataset with ground-truth comparisons.</div></div>', unsafe_allow_html=True)

    metrics = compute_metrics()

    # ── Top KPIs ───────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-val blue">{metrics['total_documents']}</div><div class="kpi-lbl">Documents Tested</div></div>
  <div class="kpi"><div class="kpi-val green">{metrics['overall_accuracy']*100:.1f}%</div><div class="kpi-lbl">Overall Accuracy</div></div>
  <div class="kpi"><div class="kpi-val blue">{metrics['macro_precision']*100:.1f}%</div><div class="kpi-lbl">Macro Precision</div></div>
  <div class="kpi"><div class="kpi-val blue">{metrics['macro_f1']*100:.1f}%</div><div class="kpi-lbl">Macro F1 Score</div></div>
</div>""", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(metrics["field_metrics"])

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Precision", x=df["field"], y=df["precision"], marker_color="#3b82f6"))
        fig.add_trace(go.Bar(name="Recall",    x=df["field"], y=df["recall"],    marker_color="#10b981"))
        fig.add_trace(go.Bar(name="F1 Score",  x=df["field"], y=df["f1"],        marker_color="#8b5cf6"))
        fig.update_layout(
            title="Precision / Recall / F1 by Field",
            barmode="group",
            height=340,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", size=11, color="#374151"),
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=0, r=0, t=40, b=60),
            yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="#f1f5f9"),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df, x="accuracy", y="field", orientation="h",
            title="Field-level Accuracy",
            color="accuracy",
            color_continuous_scale=["#bfdbfe","#2563eb"],
            range_color=[0, 1],
            labels={"accuracy": "Accuracy", "field": ""},
        )
        fig2.update_layout(
            height=340,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", size=11, color="#374151"),
            margin=dict(l=0, r=0, t=40, b=20),
            coloraxis_showscale=False,
            xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Radar chart ────────────────────────────────────────────────────────────
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(r=df["f1"].tolist() + [df["f1"].iloc[0]], theta=df["field"].tolist() + [df["field"].iloc[0]], fill="toself", name="F1 Score", line_color="#3b82f6", fillcolor="rgba(59,130,246,.15)"))
    fig3.add_trace(go.Scatterpolar(r=df["precision"].tolist() + [df["precision"].iloc[0]], theta=df["field"].tolist() + [df["field"].iloc[0]], fill="toself", name="Precision", line_color="#10b981", fillcolor="rgba(16,185,129,.1)"))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Field Coverage Radar", height=360,
        paper_bgcolor="white",
        font=dict(family="Inter", size=11, color="#374151"),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=20, r=20, t=48, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Field metrics table ────────────────────────────────────────────────────
    st.markdown('<div style="font-weight:700;color:#0f172a;margin:8px 0 12px;">📋 Field-level Metrics Table</div>', unsafe_allow_html=True)
    table_df = df[["field","precision","recall","f1","accuracy","tp","fp","fn"]].copy()
    table_df.columns = ["Field","Precision","Recall","F1","Accuracy","TP","FP","FN"]
    table_df[["Precision","Recall","F1","Accuracy"]] = table_df[["Precision","Recall","F1","Accuracy"]].applymap(lambda x: f"{x*100:.1f}%")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # ── Ground Truth vs Predicted ──────────────────────────────────────────────
    st.markdown('<div style="font-weight:700;color:#0f172a;margin:20px 0 12px;">🔍 Ground Truth vs Predicted</div>', unsafe_allow_html=True)

    for sample in TEST_DATASET:
        with st.expander(f"📄 {sample['doc_id']}", expanded=False):
            gt, pred = sample["ground_truth"], sample["predicted"]
            rows = ""
            for field in gt:
                gt_v  = gt.get(field, "—")
                pr_v  = pred.get(field, "—")
                match = gt_v.lower().strip() in pr_v.lower().strip() or pr_v.lower().strip() in gt_v.lower().strip()
                icon  = "✅" if match else "❌"
                rows += f"<tr><td style='color:#64748b;font-weight:500;'>{field.replace('_',' ').title()}</td><td style='color:#15803d;'>{gt_v}</td><td style='color:#1d4ed8;'>{pr_v}</td><td style='text-align:center;'>{icon}</td></tr>"
            st.markdown(f"""
<table class="tbl">
  <thead><tr><th>Field</th><th>Ground Truth</th><th>Predicted</th><th>Match</th></tr></thead>
  <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)

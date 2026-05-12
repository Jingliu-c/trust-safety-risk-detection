import streamlit as st
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Trust & Safety Risk Dashboard",
    layout="wide"
)

st.title("Trust & Safety Content Risk Dashboard")
st.caption("Model performance, enforcement thresholds, and error review for harmful content detection.")

# Load data
df = pd.read_csv("outputs/processed_comments.csv")
results_df = pd.read_csv("outputs/model_predictions.csv")
threshold_df = pd.read_csv("outputs/threshold_analysis.csv")
false_pos = pd.read_csv("outputs/false_positives.csv")
false_neg = pd.read_csv("outputs/false_negatives.csv")

# Sidebar
st.sidebar.header("Controls")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

action_filter = st.sidebar.multiselect(
    "Recommended Action",
    options=["Auto Remove", "Human Review", "Monitor", "Allow"],
    default=["Auto Remove", "Human Review", "Monitor", "Allow"]
)

# Dynamic prediction
results_df["dynamic_pred"] = (results_df["risk_score"] >= threshold).astype(int)

def assign_action(score):
    if score >= 0.90:
        return "Auto Remove"
    elif score >= 0.50:
        return "Human Review"
    elif score >= 0.20:
        return "Monitor"
    else:
        return "Allow"

results_df["dynamic_action"] = results_df["risk_score"].apply(assign_action)

filtered = results_df[results_df["dynamic_action"].isin(action_filter)]

# Metrics
precision = precision_score(results_df["actual_label"], results_df["dynamic_pred"], zero_division=0)
recall = recall_score(results_df["actual_label"], results_df["dynamic_pred"], zero_division=0)
f1 = f1_score(results_df["actual_label"], results_df["dynamic_pred"], zero_division=0)

fp_count = ((results_df["actual_label"] == 0) & (results_df["dynamic_pred"] == 1)).sum()
fn_count = ((results_df["actual_label"] == 1) & (results_df["dynamic_pred"] == 0)).sum()
flagged = results_df["dynamic_pred"].sum()

st.header("Executive Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Comments", len(results_df))
c2.metric("Flagged Comments", int(flagged))
c3.metric("Precision", round(precision, 3))
c4.metric("Recall", round(recall, 3))

c5, c6, c7 = st.columns(3)
c5.metric("F1 Score", round(f1, 3))
c6.metric("False Positives", int(fp_count))
c7.metric("False Negatives", int(fn_count))

st.divider()

# Threshold analysis
st.header("Precision / Recall Tradeoff")

st.write(
    "Lower thresholds improve recall but create more false positives. "
    "Higher thresholds improve precision but miss more harmful content."
)

st.line_chart(
    threshold_df.set_index("threshold")[["precision", "recall", "f1"]]
)

st.dataframe(threshold_df, use_container_width=True)

st.divider()

# Enforcement action breakdown
st.header("Enforcement Action Breakdown")

action_counts = results_df["dynamic_action"].value_counts().reindex(
    ["Auto Remove", "Human Review", "Monitor", "Allow"]
).fillna(0)

st.bar_chart(action_counts)

st.divider()

# Abuse label distribution
st.header("Abuse Category Distribution")

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

label_rates = df[label_cols].mean().sort_values(ascending=False)
st.bar_chart(label_rates)

st.divider()

# Risk score distribution
st.header("Risk Score Distribution")

score_bins = pd.cut(
    results_df["risk_score"],
    bins=[0, 0.2, 0.5, 0.9, 1.0],
    labels=["Allow (<0.2)", "Monitor (0.2-0.5)", "Human Review (0.5-0.9)", "Auto Remove (>=0.9)"],
    include_lowest=True
)

score_dist = score_bins.value_counts().sort_index()
st.bar_chart(score_dist)

st.divider()

# Error analysis
st.header("Error Review")

tab1, tab2, tab3 = st.tabs([
    "False Positives",
    "False Negatives",
    "Sample Enforcement Queue"
])

with tab1:
    st.subheader("False Positives: Potential Over-Enforcement")
    st.write(
        "These are comments the model flagged as harmful, but the actual label is not harmful. "
        "This matters because excessive false positives may reduce user trust."
    )
    st.dataframe(
        false_pos[["comment_text", "risk_score", "recommended_action"]].head(20),
        use_container_width=True
    )

with tab2:
    st.subheader("False Negatives: Missed Harmful Content")
    st.write(
        "These are harmful comments the model failed to flag at the default threshold. "
        "This is a key Trust & Safety risk because harmful content may remain visible."
    )
    st.dataframe(
        false_neg[["comment_text", "risk_score", "recommended_action"]].head(20),
        use_container_width=True
    )

with tab3:
    st.subheader("Filtered Enforcement Queue")
    st.dataframe(
        filtered[["comment_text", "actual_label", "risk_score", "dynamic_action"]]
        .sort_values("risk_score", ascending=False)
        .head(100),
        use_container_width=True
    )

st.divider()

# Recommendation
st.header("Business Recommendation")

st.markdown("""
**Recommended enforcement strategy:**

- Use **risk score >= 0.90** for high-confidence auto-removal because precision is very high.
- Route **0.50–0.90** cases to human review to reduce over-enforcement risk.
- Monitor **0.20–0.50** cases because lowering the threshold materially improves recall.
- For high-severity categories such as threats or identity hate, prioritize recall and route borderline cases to review.

This design treats the model as a decision-support system rather than a fully automated enforcement tool.
""")
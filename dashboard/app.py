import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Trust & Safety Risk Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("outputs/processed_comments.csv")
    results_df = pd.read_csv("outputs/model_predictions.csv")
    threshold_df = pd.read_csv("outputs/threshold_analysis.csv")
    false_pos = pd.read_csv("outputs/false_positives.csv")
    false_neg = pd.read_csv("outputs/false_negatives.csv")
    return df, results_df, threshold_df, false_pos, false_neg

df, results_df, threshold_df, false_pos, false_neg = load_data()

# -----------------------------
# Helper functions
# -----------------------------
def assign_action(score):
    if score >= 0.90:
        return "Auto Remove"
    elif score >= 0.50:
        return "Human Review"
    elif score >= 0.20:
        return "Monitor"
    else:
        return "Allow"

label_cols = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛡️ Trust & Safety Controls")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05
)

selected_actions = st.sidebar.multiselect(
    "Filter Enforcement Actions",
    ["Auto Remove", "Human Review", "Monitor", "Allow"],
    default=["Auto Remove", "Human Review", "Monitor", "Allow"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard simulates a Trust & Safety decision-support system "
    "for harmful content detection and moderation review."
)

# -----------------------------
# Dynamic calculations
# -----------------------------
results_df["dynamic_pred"] = (results_df["risk_score"] >= threshold).astype(int)
results_df["dynamic_action"] = results_df["risk_score"].apply(assign_action)

filtered_df = results_df[results_df["dynamic_action"].isin(selected_actions)]

precision = precision_score(
    results_df["actual_label"],
    results_df["dynamic_pred"],
    zero_division=0
)

recall = recall_score(
    results_df["actual_label"],
    results_df["dynamic_pred"],
    zero_division=0
)

f1 = f1_score(
    results_df["actual_label"],
    results_df["dynamic_pred"],
    zero_division=0
)

false_positive_count = (
    (results_df["actual_label"] == 0) &
    (results_df["dynamic_pred"] == 1)
).sum()

false_negative_count = (
    (results_df["actual_label"] == 1) &
    (results_df["dynamic_pred"] == 0)
).sum()

flagged_count = results_df["dynamic_pred"].sum()
harmful_rate = df["is_harmful"].mean()

# -----------------------------
# Header
# -----------------------------
st.title("Trust & Safety Content Risk Dashboard")
st.caption(
    "Interactive dashboard for harmful content detection, model evaluation, "
    "threshold tuning, and enforcement decision support."
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "Threshold Analysis",
    "Enforcement Queue",
    "Error Review",
    "Business Recommendation"
])

# -----------------------------
# Tab 1: Executive Overview
# -----------------------------
with tab1:
    st.subheader("Executive Overview")

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    kpi1.metric("Total Comments", f"{len(results_df):,}")
    kpi2.metric("Harmful Rate", f"{harmful_rate:.1%}")
    kpi3.metric("Precision", f"{precision:.1%}")
    kpi4.metric("Recall", f"{recall:.1%}")
    kpi5.metric("False Negatives", f"{false_negative_count:,}")
    kpi6.metric("Flagged Comments", f"{flagged_count:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Abuse Category Distribution")
        label_rates = (
            df[label_cols]
            .mean()
            .reset_index()
            .rename(columns={"index": "Category", 0: "Rate"})
        )

        fig = px.bar(
            label_rates,
            x="Category",
            y="Rate",
            title="Share of Comments by Abuse Category",
            text=label_rates["Rate"].apply(lambda x: f"{x:.1%}")
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Risk Score Distribution")
        fig = px.histogram(
            results_df,
            x="risk_score",
            nbins=40,
            title="Distribution of Model Risk Scores"
        )
        fig.update_layout(xaxis_title="Risk Score", yaxis_title="Comment Count")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 2: Threshold Analysis
# -----------------------------
with tab2:
    st.subheader("Precision / Recall Tradeoff")

    st.info(
        "Lower thresholds increase recall but may create more false positives. "
        "Higher thresholds increase precision but may miss harmful content."
    )

    fig = px.line(
        threshold_df,
        x="threshold",
        y=["precision", "recall", "f1"],
        markers=True,
        title="Model Performance by Decision Threshold"
    )
    fig.update_layout(
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis_tickformat=".0%"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Threshold Results")
    st.dataframe(
        threshold_df.style.format({
            "precision": "{:.1%}",
            "recall": "{:.1%}",
            "f1": "{:.1%}",
            "threshold": "{:.2f}"
        }),
        use_container_width=True
    )

# -----------------------------
# Tab 3: Enforcement Queue
# -----------------------------
with tab3:
    st.subheader("Enforcement Queue")

    action_counts = (
        results_df["dynamic_action"]
        .value_counts()
        .reindex(["Auto Remove", "Human Review", "Monitor", "Allow"])
        .fillna(0)
        .reset_index()
    )
    action_counts.columns = ["Action", "Count"]

    fig = px.bar(
        action_counts,
        x="Action",
        y="Count",
        title="Recommended Enforcement Actions"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Review Queue")
    st.dataframe(
        filtered_df[
            ["comment_text", "actual_label", "risk_score", "dynamic_action"]
        ]
        .sort_values("risk_score", ascending=False)
        .head(100),
        use_container_width=True
    )

# -----------------------------
# Tab 4: Error Review
# -----------------------------
with tab4:
    st.subheader("Error Review")

    st.markdown(
        """
        Error analysis is critical for Trust & Safety systems because different
        error types create different risks:
        
        - **False positives** may cause over-enforcement and reduce user trust.
        - **False negatives** may allow harmful content to remain visible.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### False Positives")
        st.metric("Potential Over-Enforcement", f"{len(false_pos):,}")
        st.dataframe(
            false_pos[["comment_text", "risk_score", "recommended_action"]]
            .sort_values("risk_score", ascending=False)
            .head(25),
            use_container_width=True
        )

    with col2:
        st.markdown("### False Negatives")
        st.metric("Missed Harmful Content", f"{len(false_neg):,}")
        st.dataframe(
            false_neg[["comment_text", "risk_score", "recommended_action"]]
            .sort_values("risk_score", ascending=False)
            .head(25),
            use_container_width=True
        )

# -----------------------------
# Tab 5: Business Recommendation
# -----------------------------
with tab5:
    st.subheader("Business Recommendation")

    st.markdown(
        """
        ### Recommended Enforcement Strategy

        Based on the threshold analysis and error review, this system should be used as a
        **decision-support tool**, not a fully automated enforcement system.

        #### Proposed Policy

        | Risk Score | Recommended Action | Rationale |
        |---|---|---|
        | >= 0.90 | Auto Remove | Very high precision; suitable for high-confidence enforcement |
        | 0.50 – 0.90 | Human Review | Borderline cases require human judgment |
        | 0.20 – 0.50 | Monitor | Useful for recall improvement and trend monitoring |
        | < 0.20 | Allow | Low model risk score |

        #### Key Insight

        The baseline model is conservative. It achieves strong precision, but recall is
        limited, meaning some harmful content is missed. For severe harm categories such
        as threats or identity hate, the system should prioritize recall and route more
        borderline cases to human review.
        """
    )

    st.success(
        "Portfolio takeaway: This dashboard demonstrates model evaluation, "
        "threshold tuning, error analysis, and Trust & Safety decision design."
    )

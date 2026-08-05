import random
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agentic10 import AssessmentPipeline, DEMO_BUSINESS, train_or_load_model
from synthetic5 import sector_config
#from synthetic6 import sector_config

RISK_BAND_COLORS = {
    "High Risk": "#e74c3c",
    "Moderate Risk": "#f39c12",
    "Low Risk": "#2ecc71",
    "Excellent": "#27ae60",
}
RISK_BAND_STEPS = [
    (300, 499, "#f5b7b1"),
    (500, 649, "#fad7a0"),
    (650, 799, "#a9dfbf"),
    (800, 900, "#7dcea0"),
]

st.set_page_config(
    page_title="MSME Financial Health Assessment",
    page_icon="\U0001F4CA",
    layout="wide",
)

CREDIT_HISTORY_OPTIONS = ["None", "Limited", "Established"]
BUSINESS_SIZE_OPTIONS = ["Micro", "Small", "Medium"]



# ----------------------------------------------------------------------
# Model loading (cached so training only happens once per server process)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading / training the health-score model...")
def get_pipeline():
    model, model_columns, reference_stats = train_or_load_model()
    return AssessmentPipeline(model, model_columns, reference_stats)


# ----------------------------------------------------------------------
# Chart builders — one function per visual, each defensive against a
# missing data source (evidence can be legitimately absent).
# ----------------------------------------------------------------------
def health_score_gauge(ctx):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ctx.health_score,
        number={"suffix": " / 900"},
        title={"text": f"Health Score — {ctx.risk_band}"},
        gauge={
            "axis": {"range": [300, 900]},
            "bar": {"color": RISK_BAND_COLORS.get(ctx.risk_band, "#3498db")},
            "steps": [
                {"range": [lo, hi], "color": color}
                for lo, hi, color in RISK_BAND_STEPS
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def evidence_sources_pie(ctx):
    labels = ["Available", "Missing"]
    values = [len(ctx.available_sources), len(ctx.missing_sources)]
    if sum(values) == 0:
        return None
    fig = px.pie(
        names=labels, values=values, title="Evidence Coverage",
        color=labels, color_discrete_map={"Available": "#2ecc71", "Missing": "#e74c3c"},
        hole=0.4,
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def revenue_trend_line(ctx):
    if ctx.gst_data is None or ctx.gst_data.empty:
        return None
    df = ctx.gst_data[["month", "revenue", "gst_amount"]].copy()
    df["month"] = pd.to_datetime(df["month"])
    fig = px.line(
        df, x="month", y=["revenue", "gst_amount"], markers=True,
        title="Monthly Revenue & GST Liability",
        labels={"value": "₹", "month": "Month", "variable": "Metric"},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def payment_mix_bar(ctx):
    if ctx.upi_data is None or ctx.aa_data is None:
        return None
    df = pd.DataFrame({
        "month": pd.to_datetime(ctx.aa_data["month"]),
        "UPI": ctx.upi_data["upi_value"].values,
        "Cash": ctx.aa_data["cash_value"].values,
        "Bank Transfer": ctx.aa_data["bank_credits"].values,
    })
    melted = df.melt(id_vars="month", var_name="Channel", value_name="Value")
    fig = px.bar(
        melted, x="month", y="Value", color="Channel", barmode="stack",
        title="Monthly Payment Channel Mix",
        labels={"Value": "₹", "month": "Month"},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def bank_balance_line(ctx):
    if ctx.aa_data is None or ctx.aa_data.empty:
        return None
    df = ctx.aa_data[["month", "bank_balance", "bank_debits"]].copy()
    df["month"] = pd.to_datetime(df["month"])
    fig = px.line(
        df, x="month", y=["bank_balance", "bank_debits"], markers=True,
        title="Bank Balance vs Monthly Debits",
        labels={"value": "₹", "month": "Month", "variable": "Metric"},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def epfo_bar(ctx):
    if ctx.epfo_data is None or ctx.epfo_data.empty:
        return None
    df = ctx.epfo_data[["month", "employees", "salary_paid"]].copy()
    df["month"] = pd.to_datetime(df["month"])
    fig = px.bar(
        df, x="month", y="employees", title="Monthly Headcount (EPFO)",
        labels={"employees": "Employees", "month": "Month"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def recommendations_bar(ctx):
    if not ctx.recommendations:
        return None
    df = pd.DataFrame(ctx.recommendations)
    if "priority" not in df.columns:
        return None
    counts = df["priority"].value_counts().reset_index()
    counts.columns = ["Priority", "Count"]
    fig = px.bar(
        counts, x="Priority", y="Count", title="Recommendations by Priority",
        color="Priority",
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def render_charts(ctx):
    st.subheader("Overview")

    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(health_score_gauge(ctx), width='stretch')
    with g2:
        pie = evidence_sources_pie(ctx)
        if pie is not None:
            st.plotly_chart(pie, width='stretch')

    rec_chart = recommendations_bar(ctx)
    if rec_chart is not None:
        st.plotly_chart(rec_chart, width='stretch')

    for chart in (revenue_trend_line(ctx), payment_mix_bar(ctx), bank_balance_line(ctx), epfo_bar(ctx)):
        if chart is not None:
            st.plotly_chart(chart, width='stretch')


# ----------------------------------------------------------------------
# Shared rendering for a completed assessment
# ----------------------------------------------------------------------
def render_result(ctx, business):
    if not ctx.business_verified:
        st.error("Business verification failed — the assessment could not proceed.")
        for err in ctx.verification_errors:
            st.write(f"- {err}")
        return

    st.success("Assessment completed.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Health Score", f"{ctx.health_score:.0f} / 900")
    m2.metric("Risk Band", ctx.risk_band)
    m3.metric("Confidence", f"{ctx.confidence:.0%}")

    render_charts(ctx)

    st.subheader("Evidence Sources")
    e1, e2 = st.columns(2)
    with e1:
        st.write("**Available**")
        for s in sorted(ctx.available_sources):
            st.write(f"✅ {s}")
    with e2:
        st.write("**Missing**")
        if ctx.missing_sources:
            for s in ctx.missing_sources:
                st.write(f"⚠️ {s}")
        else:
            st.write("None — all four data sources were collected.")

    st.subheader("Explanation")
    if ctx.explanation:
        st.text(ctx.explanation)
    else:
        st.write("No explanation generated.")

    st.subheader("Recommendations")
    if ctx.recommendations:
        st.dataframe(pd.DataFrame(ctx.recommendations), width='stretch')
        if ctx.primary_recommendation:
            st.info(f"Primary recommendation: **{ctx.primary_recommendation}**")
    else:
        st.write("No recommendations generated.")

    st.subheader("AI Credit Officer Narrative")

    if ctx.llm_summary:
        if ctx.llm_summary_status == "Generated":
            with st.chat_message("assistant"):
                st.write(ctx.llm_summary)
        else:
            st.warning(ctx.llm_summary)
    else:
        st.write("No LLM narrative generated.")

    if getattr(ctx, "llm_prompt", None):
        with st.expander("Prompt sent to the LLM"):
            st.text(ctx.llm_prompt)

    with st.expander("Simulated data sources (auto-generated by the pipeline)"):
        t1, t2, t3, t4 = st.tabs(["GST", "UPI", "Account Aggregator", "EPFO"])
        with t1:
            if ctx.gst_data is not None:
                fig = px.line(ctx.gst_data, x="month", y="gst_amount", markers=True,
                               title="Monthly GST Amount")
                st.plotly_chart(fig, width='stretch')
            st.dataframe(ctx.gst_data, width='stretch')
        with t2:
            if ctx.upi_data is not None:
                fig = px.bar(ctx.upi_data, x="month", y="upi_count", title="Monthly UPI Transaction Count")
                st.plotly_chart(fig, width='stretch')
            st.dataframe(ctx.upi_data, width='stretch')
        with t3:
            if ctx.aa_data is not None:
                fig = px.line(ctx.aa_data, x="month", y="bank_balance", markers=True,
                               title="Monthly Bank Balance")
                st.plotly_chart(fig, width='stretch')
            st.dataframe(ctx.aa_data, width='stretch')
        with t4:
            if ctx.epfo_data is not None:
                fig = px.bar(ctx.epfo_data, x="month", y="epfo_contribution", title="Monthly EPFO Contribution")
                st.plotly_chart(fig, width='stretch')
            st.dataframe(ctx.epfo_data, width='stretch')

    with st.expander("Agent trace"):
        st.code("\n".join(ctx.log), language=None)

    with st.expander("Decision history"):
        hist_df = pd.DataFrame(ctx.decision_history)
        for col in ("reasoning", "result"):
            if col in hist_df.columns:
                hist_df[col] = hist_df[col].astype(str)
        st.dataframe(hist_df, width='stretch')

    with st.expander("Business profile sent to the pipeline"):
        st.json(business)


def run_and_render(business):
    pipeline = get_pipeline()
    with st.spinner("Running the agentic assessment pipeline..."):
        ctx = pipeline.assess(business)
    render_result(ctx, business)


st.title("\U0001F4CA MSME Financial Health Assessment")
st.caption(
    "Business details go in \u2192 the pipeline's agents simulate GST, UPI, "
    "Account Aggregator (bank) and EPFO data, run the health-score model, "
    "and then a local LLM agent (Llama 3.2 via Ollama) turns all of that "
    "into a professional credit narrative \u2192 out comes a health score, "
    "risk band, explanation and recommendations, plus an AI-written summary."
)

tab_demo, tab_custom = st.tabs(["\U0001F9EA Demo Business ", "\U0001F4DD Custom Business "])

# ---------------- Tab 1: hardcoded demo business ----------------
with tab_demo:

    st.json(DEMO_BUSINESS)

    if st.button("Run Assessment on Demo Business", type="primary"):
        run_and_render(DEMO_BUSINESS)

# ---------------- Tab 2: custom user-entered business ----------------
with tab_custom:
    st.write("Fill in your business's details below. Nothing here is data-sourced from "
             "GST, UPI, AA or EPFO directly — the pipeline generates all of that on its "
             "own once you submit.")

    # Sector lives outside the form so its subcategory list updates live —
    # widgets inside st.form only rerun the app on submit.
    sector = st.selectbox("Sector", list(sector_config.keys()))

    with st.form("custom_business_form"):
        col1, col2 = st.columns(2)

        with col1:
            business_id = st.number_input(
                "Business ID", min_value=1, value=random.randint(100000, 999999), step=1,
                help="Used internally to make the simulated data reproducible for this business."
            )
            subcategory = st.selectbox(
                "Subcategory", list(sector_config[sector]["subcategories"].keys())
            )
            business_size = st.selectbox("Business Size", BUSINESS_SIZE_OPTIONS, index=1)
            
            age_months = st.number_input(
                "Business Age (months)", min_value=1, max_value=1200, value=24
            )

        with col2:
            gst_registered = st.checkbox("GST Registered", value=True)
            pan_available = st.checkbox("PAN Available", value=True)
            udyam_registered = st.checkbox("Udyam (MSME) Registered", value=True)
            credit_history = st.selectbox("Credit History", CREDIT_HISTORY_OPTIONS, index=1)
            has_relationship_our_bank = st.checkbox("Existing relationship with our bank", value=False)
            has_relationship_other_bank = st.checkbox("Existing relationship with another bank", value=True)

        submitted = st.form_submit_button("Run Assessment", type="primary")

    if submitted:
        custom_business = {
            "business_id": int(business_id),
            "sector": sector,
            "subcategory": subcategory,
            "business_size": business_size,
            "age_months": int(age_months),
            "gst_registered": gst_registered,
            "pan_available": pan_available,
            "udyam_registered": udyam_registered,
            "credit_history": credit_history,
            "has_relationship_our_bank": has_relationship_our_bank,
            "has_relationship_other_bank": has_relationship_other_bank,
        }
        run_and_render(custom_business)
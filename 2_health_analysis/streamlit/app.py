"""
Blood Report Health Analysis — Streamlit UI
--------------------------------------------
UI wrapper around the exact 2-stage pipeline in `health_analysis.ipynb`:

  Stage 1 (Extraction):
    - Sends the raw blood report to Gemini
    - Extracts every test value and classifies it HIGH / LOW / NORMAL
      against the reference ranges in the report

  Stage 2 (Health Summary & Sri Lankan Diet Plan):
    - Takes the Stage 1 extracted values
    - Produces a short plain-language health summary
    - Produces a Sri Lankan diet plan (Foods to avoid / Foods to eat more of)

Run with:
    pip install streamlit langchain-google-genai python-dotenv
    streamlit run streamlit_app.py

Make sure a `.env` file (same folder, or any parent folder) contains:
    GOOGLE_API_KEY=your_key_here
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()

MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(
    page_title="Blood Report Analyzer",
    page_icon="🩸",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini chat model once and cache it across reruns."""
    return ChatGoogleGenerativeAI(model=MODEL_NAME)


def run_extraction(blood_report: str) -> str:
    """Stage 1: extract all test values and classify HIGH/LOW/NORMAL."""
    llm = get_llm()
    extraction_prompt = f"""
    You are a medical data extraction assistant.

    From the blood report below, extract ALL test values and classify each one as HIGH, LOW, NORMAL
    based on the reference ranges provided in the report.

    Format your response as:
    -Test Name: value | Status : HIGH/LOW/NORMAL | Reference : range

    Blood Report:
    {blood_report}
"""
    extraction_response = llm.invoke(extraction_prompt)
    return extraction_response.text


def run_diet_plan(extracted_values: str) -> str:
    """Stage 2: health summary + Sri Lankan diet plan from the extracted values."""
    llm = get_llm()
    diet_prompt = f""" 
    You are a clinical nutritionist specializing in Sri Lankan dietary habits.

    Based on the blood work analysis below, write:
    1. A short health summary in 4-5 lines explaining the patient's condition in simple language
    2. A short, practical Sri Lankan diet plan having only two section (1) Foods to avoid (2) Foods to eat more of.
    Do not include any other sections in diet plan

    Blood work Analysis:
    {extracted_values}
"""
    diet_response = llm.invoke(diet_prompt)
    return diet_response.text


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = None
if "diet_plan" not in st.session_state:
    st.session_state.diet_plan = None

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("Blood Report Health Analyzer")
st.caption(f"Powered by `{MODEL_NAME}` · Educational use only, not medical advice.")
st.write(
    "Upload a blood test report (as `.txt`) or paste the text below. "
    "Stage 1 extracts and classifies every test value, then Stage 2 turns that "
    "into a plain-language summary and a Sri Lankan diet plan."
)

tab_upload, tab_paste = st.tabs(["📁 Upload file", "📋 Paste text"])

blood_report = ""

with tab_upload:
    uploaded_file = st.file_uploader("Choose a .txt blood report", type=["txt"])
    if uploaded_file is not None:
        blood_report = uploaded_file.read().decode("utf-8", errors="ignore")
        with st.expander("Preview uploaded report", expanded=False):
            st.text(blood_report[:2000])

with tab_paste:
    pasted_text = st.text_area(
        "Paste the raw blood report text here",
        height=250,
        placeholder="e.g. Hemoglobin: 13.5 g/dL (Normal: 13.0-17.0) ...",
    )
    if pasted_text.strip():
        blood_report = pasted_text

st.divider()

col1, col2 = st.columns([1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze Report", type="primary", use_container_width=True)
with col2:
    if not blood_report.strip():
        st.info("Provide a report via upload or paste before analyzing.")

if analyze_clicked:
    if not blood_report.strip():
        st.warning("Please upload or paste a blood report first.")
    else:
        try:
            with st.spinner("Stage 1 — Extracting and classifying values..."):
                st.session_state.extracted_values = run_extraction(blood_report)

            with st.spinner("Stage 2 — Generating health summary & diet plan..."):
                st.session_state.diet_plan = run_diet_plan(st.session_state.extracted_values)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {exc}")

if st.session_state.extracted_values:
    st.subheader("🧪 Stage 1 — Extracted Values")
    st.code(st.session_state.extracted_values, language=None)

if st.session_state.diet_plan:
    st.subheader("🥗 Stage 2 — Health Summary & Sri Lankan Diet Plan")
    st.markdown(st.session_state.diet_plan)

    combined_report = (
        "=== STAGE 1: EXTRACTED VALUES ===\n"
        f"{st.session_state.extracted_values}\n\n"
        "=== STAGE 2: HEALTH SUMMARY & DIET PLAN ===\n"
        f"{st.session_state.diet_plan}"
    )
    st.download_button(
        "⬇️ Download full report (.md)",
        data=combined_report,
        file_name="blood_report_analysis.md",
        mime="text/markdown",
    )
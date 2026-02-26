import streamlit as st
from groq import Groq
import re

# -------- CONFIG --------
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

st.set_page_config(page_title="Med-Dev", layout="wide")

# -------- CUSTOM STYLING --------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #f8fafc;
}
            
            [data-testid="stAppViewContainer"] {
    background: transparent !important;
}

h1 { color: #38bdf8; text-align: center; }
h2, h3 { color: #60a5fa; }

.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 15px rgba(56,189,248,0.6);
}

textarea, input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 8px !important;
}

.output-card {
    background: #1e293b;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
    color: #ffffff !important;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------- SIDEBAR --------
with st.sidebar:
    st.title("Med-Dev v1")
    st.markdown("Structured Clinical Reasoning Assistant")
    st.markdown("---")
    st.markdown("For educational use only.")
    st.markdown("Not for real patient management.")

# -------- HEADER --------
st.markdown("""
<div style="display:flex; justify-content:center; align-items:center; gap:15px;">
    <span class="pulse">🩺</span>
    <h1 style="margin:0; color:#38bdf8;">Med-Dev</h1>
</div>

<style>
.pulse {
    display: inline-block;
    font-size: 40px;
    animation: pulseAnim 1.5s infinite;
}

@keyframes pulseAnim {
    0% { transform: scale(1); }
    50% { transform: scale(1.3); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# -------- INPUT SECTION --------
st.subheader("Clinical Input")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])

with col2:
    vitals = st.text_area("Vitals (HR, BP, Temp, SpO2)")
    labs = st.text_area("Lab Values")

symptoms = st.text_area("Symptoms / Presenting Complaints")

st.markdown("---")

# -------- GENERATE BUTTON --------
if st.button("Generate Clinical Analysis", use_container_width=True):

    case_input = f"""
Age: {age}
Sex: {sex}
Symptoms: {symptoms}
Vitals: {vitals}
Lab Values: {labs}
"""

    prompt = f"""
Provide a structured clinical reasoning summary using exactly the following headings:

PROBLEM REPRESENTATION - content
DOMINANT SYNDROME - content
TOP 3 DIFFERENTIALS - content
RED FLAGS - content
BROAD MANAGEMENT PRINCIPLES - content
CRITICAL MISSING INFORMATION - content

Rules:
• Each section must be exactly one concise line.
• Exactly 3 differentials separated by commas.
• No numbering.
• No extra commentary outside the format.
• Complete all six sections.

CASE:
{case_input}
"""

    with st.spinner("Med-Dev is reasoning..."):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a senior internal medicine consultant performing structured clinical reasoning.

Integrate pathophysiology and contextual prioritization.
Avoid generic textbook lists.
Follow the requested output format strictly."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.01,
                max_tokens=800
            )

            raw_output = response.choices[0].message.content

            if not raw_output or raw_output.strip() == "":
                st.error("Model returned empty response.")
                st.stop()

            raw_output = raw_output.strip()

        except Exception as e:
            st.error(f"Actual error: {e}")
            st.stop()

    # -------- SAFE FORMATTING --------
    formatted_output = raw_output.replace("**", "")

    formatted_output = formatted_output.replace(
        "PROBLEM REPRESENTATION",
        "<strong>PROBLEM REPRESENTATION</strong>"
    ).replace(
        "DOMINANT SYNDROME",
        "<br><br><strong>DOMINANT SYNDROME</strong>"
    ).replace(
        "TOP 3 DIFFERENTIALS",
        "<br><br><strong>TOP 3 DIFFERENTIALS</strong>"
    ).replace(
        "RED FLAGS",
        "<br><br><strong>RED FLAGS</strong>"
    ).replace(
        "BROAD MANAGEMENT PRINCIPLES",
        "<br><br><strong>BROAD MANAGEMENT PRINCIPLES</strong>"
    ).replace(
        "CRITICAL MISSING INFORMATION",
        "<br><br><strong>CRITICAL MISSING INFORMATION</strong>"
    )

    st.markdown("## Clinical Analysis")
    st.markdown("---")
    st.markdown(
        f"<div class='output-card'>{formatted_output}</div>",
        unsafe_allow_html=True
    )
    # -------- FOOTER --------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>Built by D3V</div>",
    unsafe_allow_html=True
)
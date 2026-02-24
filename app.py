import streamlit as st
from huggingface_hub import InferenceClient
import re

# -------- CONFIG --------
import os
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

client = InferenceClient(
    model=MODEL_NAME,
    token=HF_TOKEN
)

st.set_page_config(page_title="Med-Dev", layout="wide")

# -------- CUSTOM STYLING --------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #f8fafc;
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
Provide a concise structured clinical reasoning summary.

Strict Format:
Each section must appear on ONE line only.
No numbering.
No extra commentary.
Exactly 3 differentials.
All headings in uppercase.
Format exactly as:

PROBLEM REPRESENTATION - content
DOMINANT SYNDROME - content
TOP 3 DIFFERENTIALS - content
RED FLAGS - content
BROAD MANAGEMENT PRINCIPLES - content
CRITICAL MISSING INFORMATION - content

CASE:
{case_input}
"""

    with st.spinner("Med-Dev is reasoning..."):
        try:
            response = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a structured clinical reasoning engine. Follow the requested format exactly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.02,
                max_tokens=600
            )

            raw_output = response.choices[0].message.content

        except Exception:
            st.error("Model temporarily unavailable. Please retry.")
            st.stop()

    output = raw_output.replace("**", "")

    # Bold the headings
    formatted_output = output.replace("PROBLEM REPRESENTATION", "**PROBLEM REPRESENTATION**") \
                             .replace("DOMINANT SYNDROME", "**DOMINANT SYNDROME**") \
                             .replace("TOP 3 DIFFERENTIALS", "**TOP 3 DIFFERENTIALS**") \
                             .replace("RED FLAGS", "**RED FLAGS**") \
                             .replace("BROAD MANAGEMENT PRINCIPLES", "**BROAD MANAGEMENT PRINCIPLES**") \
                             .replace("CRITICAL MISSING INFORMATION", "**CRITICAL MISSING INFORMATION**")

    st.markdown("## Clinical Analysis")
    st.markdown("---")
    st.markdown(f"<div class='output-card'>{formatted_output}</div>", unsafe_allow_html=True)
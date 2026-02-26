import html
import re
from typing import Dict, Optional

import streamlit as st
from groq import Groq

# -------- CONFIG --------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title="Med-Dev", layout="wide")

SECTION_HEADINGS = [
    "PROBLEM REPRESENTATION",
    "DOMINANT SYNDROME",
    "TOP 3 DIFFERENTIALS",
    "RED FLAGS",
    "BROAD MANAGEMENT PRINCIPLES",
    "CRITICAL MISSING INFORMATION",
]

HEADINGS_PATTERN = (
    r"(PROBLEM REPRESENTATION|DOMINANT SYNDROME|TOP 3 DIFFERENTIALS|RED FLAGS|"
    r"BROAD MANAGEMENT PRINCIPLES|CRITICAL MISSING INFORMATION)\s*-\s*"
)
HEADING_SPLIT_REGEX = re.compile(HEADINGS_PATTERN, flags=re.IGNORECASE)

if "raw_output" not in st.session_state:
    st.session_state.raw_output = None
if "repair_used" not in st.session_state:
    st.session_state.repair_used = False
if "generation_error" not in st.session_state:
    st.session_state.generation_error = None
if "generation_warning" not in st.session_state:
    st.session_state.generation_warning = None


def build_system_prompt() -> str:
    return """You are a senior internal medicine consultant producing structured, high-fidelity clinical reasoning.

Behavior constraints:
- Prioritize synthesis over enumeration.
- For every major finding, connect finding -> mechanism -> syndrome/diagnosis.
- Integrate all abnormal labs explicitly, especially in DOMINANT SYNDROME.
- Avoid generic textbook lists or unanchored differentials.
- Prioritize diagnoses using Bayesian updating: pretest plausibility adjusted by case-specific discriminative evidence.
- Enforce etiologic hierarchy in differentials: root cause -> intermediate mechanism -> downstream complication.
- Use concise but complete sentences only.
- Do not output chain-of-thought.

Internal quality checks before finalizing (do not reveal):
1) All 6 required headings are present exactly once and in order.
2) No section is empty or truncated.
3) Problem representation explicitly integrates key clues and abnormal labs.
4) TOP 3 DIFFERENTIALS follows the exact schema with posterior probabilities, strict descending posterior order, and fixed etiologic hierarchy.
5) Each differential includes at least 3 supporting clues and at least 1 opposing clue.
6) Final output ends with a complete sentence.

Output rules:
- Return only the 6 required lines.
- No bullets, no preface, no postscript.
"""


def build_user_prompt(case_input: str) -> str:
    return f"""Generate a structured clinical reasoning summary with EXACTLY these headings in this exact order:

PROBLEM REPRESENTATION - 
DOMINANT SYNDROME - 
TOP 3 DIFFERENTIALS - 
RED FLAGS - 
BROAD MANAGEMENT PRINCIPLES - 
CRITICAL MISSING INFORMATION - 

Formatting requirements:
- Exactly 6 lines total (one line per heading).
- Each line must contain 1-3 complete sentences.
- No extra headings, no bullets, no numbering.
- End every line with sentence punctuation.

Clinical depth requirements:
- PROBLEM REPRESENTATION: concise synthesis of acuity + key positives + discriminative clues + unifying concern.
- DOMINANT SYNDROME: explicitly integrate abnormal labs and explain lab -> mechanism -> syndrome.
- TOP 3 DIFFERENTIALS: exactly 3 diagnoses in this strict single-line template:
  "Dx1: <diagnosis> [Posterior: <0.00-1.00>] [Hierarchy: root-cause] (For: <finding1>; <finding2>; <finding3> | Against: <counter-clue>) || Dx2: <diagnosis> [Posterior: <0.00-1.00>] [Hierarchy: intermediate-mechanism] (For: <finding1>; <finding2>; <finding3> | Against: <counter-clue>) || Dx3: <diagnosis> [Posterior: <0.00-1.00>] [Hierarchy: downstream-complication] (For: <finding1>; <finding2>; <finding3> | Against: <counter-clue>)"
- Posterior probabilities must be strictly descending: Dx1 > Dx2 > Dx3.
- RED FLAGS: immediate deterioration risks linked to this case.
- BROAD MANAGEMENT PRINCIPLES: stabilization, targeted diagnostics, and early risk-mitigation priorities.
- CRITICAL MISSING INFORMATION: highest-yield data that would change diagnosis or management now.

Completeness requirements:
- Internally verify all abnormal labs are integrated.
- Internally verify each differential includes >=3 supporting clues and >=1 opposing clue.
- Internally verify Bayesian ordering and etiologic hierarchy are correct.
- Internally verify no section is incomplete.
- Internally verify final word completes a sentence.

CASE DATA:
{case_input}
"""


def build_repair_prompt(case_input: str, prior_output: str) -> str:
    return f"""Rewrite the prior output into a strictly valid 6-line answer.

Rules:
- Keep exactly these 6 headings and order:
  PROBLEM REPRESENTATION -
  DOMINANT SYNDROME -
  TOP 3 DIFFERENTIALS -
  RED FLAGS -
  BROAD MANAGEMENT PRINCIPLES -
  CRITICAL MISSING INFORMATION -
- Keep concise complete sentences.
- Preserve clinical meaning; do not invent unrelated findings.
- TOP 3 DIFFERENTIALS must follow this exact schema:
  "Dx1: <diagnosis> [Posterior: <0.00-1.00>] [Hierarchy: root-cause] (For: <finding1>; <finding2>; <finding3> | Against: <counter-clue>) || Dx2: <diagnosis> [Posterior: <0.00-1.00>] [Hierarchy: intermediate-mechanism] (For: <finding1>; <finding2>; <finding3> | Against: <counter-clue>) || Dx3: <diagnosis> [Posterior: <0.00-1.00>] [Hierarchy: downstream-complication] (For: <finding1>; <finding2>; <finding3> | Against: <counter-clue>)"
- Posterior values must be strictly descending: Dx1 > Dx2 > Dx3.

CASE DATA:
{case_input}

PRIOR OUTPUT:
{prior_output}
"""


def extract_sections(raw_text: str) -> Optional[Dict[str, str]]:
    parts = re.split(HEADING_SPLIT_REGEX, raw_text.strip())
    if len(parts) < 13:
        return None

    sections: Dict[str, str] = {}
    canonical = {h.upper(): h for h in SECTION_HEADINGS}

    for i in range(1, len(parts), 2):
        heading_raw = parts[i].strip().upper()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        heading = canonical.get(heading_raw)
        if not heading:
            continue
        if heading in sections:
            return None
        sections[heading] = content

    if list(sections.keys()) != SECTION_HEADINGS:
        return None

    return sections


def is_complete_sentence(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and bool(re.search(r"[.!?]$", stripped))


def parse_posterior(raw_value: str) -> Optional[float]:
    value = raw_value.strip()
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1].strip()

    try:
        posterior = float(value)
    except ValueError:
        return None

    if is_percent:
        posterior = posterior / 100.0
    elif posterior > 1.0 and posterior <= 100.0:
        posterior = posterior / 100.0

    if posterior < 0.0 or posterior > 1.0:
        return None
    return posterior


def normalize_hierarchy(raw_value: str) -> Optional[str]:
    normalized = re.sub(r"[\s_]+", "-", raw_value.strip().lower())
    alias_map = {
        "root-cause": "root-cause",
        "rootcause": "root-cause",
        "primary-cause": "root-cause",
        "primary-etiology": "root-cause",
        "upstream-cause": "root-cause",
        "intermediate-mechanism": "intermediate-mechanism",
        "intermediary-mechanism": "intermediate-mechanism",
        "mechanistic-intermediate": "intermediate-mechanism",
        "downstream-complication": "downstream-complication",
        "downstream-effect": "downstream-complication",
        "complication": "downstream-complication",
    }
    return alias_map.get(normalized)


def split_clues(raw_value: str) -> list[str]:
    return [c.strip(" .") for c in re.split(r"[;,]", raw_value) if c.strip()]


def has_three_supported_differentials(top3_line: str) -> bool:
    blocks = [
        b.strip(" |")
        for b in re.split(r"\s*\|\|\s*|\s*\|\s*(?=Dx[23]:)", top3_line.strip())
        if b.strip()
    ]
    if len(blocks) != 3:
        return False

    pattern = re.compile(
        r"^Dx(?P<rank>[123]):\s*(?P<diagnosis>.+?)\s*"
        r"\[Posterior:\s*(?P<posterior>[0-9]+(?:\.[0-9]+)?%?)\]\s*"
        r"\[Hierarchy:\s*(?P<hierarchy>[^\]]+)\]\s*"
        r"\(For:\s*(?P<for>.+?)\s*\|\s*Against:\s*(?P<against>.+?)\)\.?$",
        flags=re.IGNORECASE,
    )
    expected_hierarchy = [
        "root-cause",
        "intermediate-mechanism",
        "downstream-complication",
    ]
    posteriors = []

    for idx, block in enumerate(blocks, start=1):
        match = pattern.match(block)
        if not match:
            return False

        rank = int(match.group("rank"))
        if rank != idx:
            return False

        hierarchy = normalize_hierarchy(match.group("hierarchy"))
        if hierarchy is None:
            return False
        if hierarchy != expected_hierarchy[idx - 1]:
            return False

        posterior = parse_posterior(match.group("posterior"))
        if posterior is None:
            return False
        posteriors.append(posterior)

        diagnosis = match.group("diagnosis").strip()
        if len(diagnosis) < 3:
            return False

        for_clues = split_clues(match.group("for"))
        if len(for_clues) < 3:
            return False

        against_clues = split_clues(match.group("against"))
        if len(against_clues) < 1:
            return False

    if not (posteriors[0] > posteriors[1] > posteriors[2]):
        return False

    return True


def is_valid_output(raw_text: str) -> bool:
    if not raw_text or not raw_text.strip():
        return False

    sections = extract_sections(raw_text)
    if not sections:
        return False

    for heading in SECTION_HEADINGS:
        content = sections.get(heading, "").strip()
        if not content:
            return False
        if not is_complete_sentence(content):
            return False

    if not has_three_supported_differentials(sections["TOP 3 DIFFERENTIALS"]):
        return False

    return True


def format_clinical_output(raw_text: str) -> str:
    sections = extract_sections(raw_text)
    if not sections:
        safe_text = html.escape(raw_text).replace("\n", "<br>")
        return f"<div class='analysis-section'>{safe_text}</div>"

    output_chunks = []
    for heading in SECTION_HEADINGS:
        safe_heading = html.escape(heading)
        safe_content = html.escape(sections[heading])
        output_chunks.append(
            f"<div class='analysis-section'><strong>{safe_heading}</strong> - {safe_content}</div>"
        )
    return "".join(output_chunks)


# -------- CUSTOM STYLING --------
st.markdown(
    """
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

.analysis-section {
    margin-bottom: 1em;
    line-height: 1.6;
}

.analysis-section:last-child {
    margin-bottom: 0;
}

footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# -------- SIDEBAR --------
with st.sidebar:
    st.title("Med-Dev v1")
    st.markdown("Structured Clinical Reasoning Assistant")
    st.markdown("---")
    st.markdown("For educational use only.")
    st.markdown("Not for real patient management.")

# -------- HEADER --------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

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
    st.session_state.raw_output = None
    st.session_state.generation_error = None
    st.session_state.generation_warning = None
    st.session_state.repair_used = False

    case_input = (
        f"Age: {age}\n"
        f"Sex: {sex}\n"
        f"Symptoms: {symptoms}\n"
        f"Vitals: {vitals}\n"
        f"Lab Values: {labs}\n"
    )

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(case_input)

    max_attempts = 4
    best_candidate = ""

    with st.spinner("Med-Dev is reasoning..."):
        try:
            for attempt in range(max_attempts):
                current_user_prompt = user_prompt
                if attempt > 0:
                    st.session_state.repair_used = True
                    current_user_prompt = (
                        f"{user_prompt}\n\n"
                        "REPAIR MODE: Your last output failed structure/depth/completeness checks. "
                        "Regenerate all 6 lines in exact format with complete sentences and no truncation."
                    )

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": current_user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2200,
                )

                candidate = ""
                finish_reason = ""
                if response and response.choices:
                    choice0 = response.choices[0]
                    finish_reason = (getattr(choice0, "finish_reason", "") or "").lower()
                    message = getattr(choice0, "message", None)
                    if message:
                        candidate = (getattr(message, "content", "") or "").strip()

                likely_truncated = finish_reason in {"length", "max_tokens"}
                if likely_truncated:
                    st.session_state.repair_used = True

                if candidate:
                    best_candidate = candidate

                if candidate and is_valid_output(candidate) and not likely_truncated:
                    st.session_state.raw_output = candidate
                    break

                if candidate and not likely_truncated:
                    st.session_state.repair_used = True
                    repair_response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": build_repair_prompt(case_input, candidate),
                            },
                        ],
                        temperature=0.0,
                        max_tokens=2400,
                    )

                    repaired_candidate = ""
                    repaired_finish = ""
                    if repair_response and repair_response.choices:
                        repair_choice = repair_response.choices[0]
                        repaired_finish = (
                            getattr(repair_choice, "finish_reason", "") or ""
                        ).lower()
                        repair_message = getattr(repair_choice, "message", None)
                        if repair_message:
                            repaired_candidate = (
                                getattr(repair_message, "content", "") or ""
                            ).strip()

                    repaired_truncated = repaired_finish in {"length", "max_tokens"}
                    if repaired_candidate:
                        best_candidate = repaired_candidate

                    if (
                        repaired_candidate
                        and is_valid_output(repaired_candidate)
                        and not repaired_truncated
                    ):
                        st.session_state.raw_output = repaired_candidate
                        break

            if not st.session_state.raw_output:
                if best_candidate:
                    st.session_state.raw_output = best_candidate
                    st.session_state.generation_warning = (
                        "Output shown, but strict Bayesian/schema enforcement was partial. "
                        "Click generate again for a stricter pass."
                    )
                else:
                    st.session_state.generation_error = (
                        "Unable to generate a complete 6-section analysis after retries. Please rerun."
                    )

        except Exception:
            st.session_state.generation_error = "Generation failed. Please retry."

# -------- OUTPUT RENDERING --------
if st.session_state.generation_error:
    st.error(st.session_state.generation_error)

if st.session_state.generation_warning:
    st.warning(st.session_state.generation_warning)

if st.session_state.raw_output:
    st.markdown("## Clinical Analysis")
    st.markdown("---")

    if st.session_state.repair_used:
        st.warning("Auto-repair was used to enforce completeness/format requirements.")

    formatted_output = format_clinical_output(st.session_state.raw_output)
    st.markdown(f"<div class='output-card'>{formatted_output}</div>", unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>Built by D3V</div>",
    unsafe_allow_html=True,
)

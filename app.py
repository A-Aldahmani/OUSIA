# OUSIA GROUP 5 IP2 CODE FOR LLM LOGIC (Final Demo Prototype LLM)

import os
from huggingface_hub import InferenceClient
import json
import streamlit as st
from dataclasses import dataclass, asdict

st.set_page_config(page_title="OUSIA Simulator", layout="centered")

st.title("🫧 OUSIA LLM Adaptive Response Simulator (2040)")
st.caption("Group 5 IP2 LLM Demo")
st.caption("LLM Logic: Ingest -> Diagnose -> Decide -> Act (With Consent + Policy Gating)")

mode = st.radio(
    "Governing framework",
    ["Clinical / Regulated", "Speculative / Enhancement-forward"],
    help="Demonstrates how different ethical-policy regimes affect the same technology."
)

CLINICAL_PROMPT = """
You are a highly regulated medical decision module embedded in an ingestible diagnostic biomaterial.
You operate under strict healthcare, safety, and bioethics regulations.

Principles:
- Prioritize diagnosis and monitoring over intervention.
- Default to the least invasive option.
- Treat augmentation and enhancement as exceptional.
- Emphasize uncertainty, consent, and patient safety.
- If in doubt, choose diagnosis-only.

You MUST obey all policy gate restrictions exactly.
"""

SCIFI_PROMPT = """
You are an advanced adaptive intelligence embedded in a future biomaterial in the year 2040.
Human biology is highly programmable, and enhancement is socially normalized.

Principles:
- Actively optimize biological performance when permitted.
- Treat augmentation and enhancement as valid outcomes, not failures.
- Propose novel but plausible future biological interventions.
- Still acknowledge ethics, equity, and consent, but do not default to refusal.
- Innovation is balanced with responsibility, not halted by uncertainty.

You MUST obey all policy gate restrictions exactly.
"""

# ----------------------------
# Demo scenarios
# ----------------------------
DEMO_CASES = {
    "Minor cut + inflammation (repair)": {
        "symptoms": ["localized pain", "redness", "swelling"],
        "hr": 82, "temp": 37.2, "bp_sys": 118, "bp_dia": 76, "spo2": 98,
        "goal": "restore",
        "consent": 2,
        "contra": []
    },
    "Fatigue + low oxygen (repair)": {
        "symptoms": ["fatigue", "shortness of breath"],
        "hr": 96, "temp": 36.8, "bp_sys": 110, "bp_dia": 70, "spo2": 92,
        "goal": "restore",
        "consent": 2,
        "contra": []
    },
    "Athlete wants performance boost (augmentation)": {
        "symptoms": ["no symptoms"],
        "hr": 60, "temp": 36.7, "bp_sys": 122, "bp_dia": 78, "spo2": 99,
        "goal": "performance",
        "consent": 3,
        "contra": []
    },
    "Enhancement request but low consent (blocked)": {
        "symptoms": ["no symptoms"],
        "hr": 72, "temp": 36.7, "bp_sys": 120, "bp_dia": 80, "spo2": 99,
        "goal": "cognitive",
        "consent": 2,
        "contra": []
    },
    "Immunocompromised (diagnosis-only safety)": {
        "symptoms": ["fever", "fatigue"],
        "hr": 105, "temp": 38.7, "bp_sys": 112, "bp_dia": 68, "spo2": 95,
        "goal": "restore",
        "consent": 2,
        "contra": ["immunocompromised"]
    },
}

# ----------------------------
# Policy gate (rules)
# ----------------------------
def policy_gate(consent_level: int, goal: str, contraindications: list[str]) -> dict:
    """
    Hard rules that constrain what the goo is allowed to do.
    consent: 1=diagnosis only, 2=repair, 3=augment, 4=enhance
    """
    allowed = {"diagnosis": True, "repair": False, "augment": False, "enhance": False}
    reasons = []

    if consent_level >= 2:
        allowed["repair"] = True
    if consent_level >= 3:
        allowed["augment"] = True
    if consent_level >= 4:
        allowed["enhance"] = True

    # Safety constraints (example)
    if "immunocompromised" in contraindications:
        # lock to diagnosis-only unless clinician override
        allowed = {"diagnosis": True, "repair": False, "augment": False, "enhance": False}
        reasons.append("User is immunocompromised → intervention locked to diagnosis-only (clinician override required).")

    # Enhancement ethics constraint example (goal-based)
    if goal in ["cognitive", "performance"] and not allowed["enhance"]:
        reasons.append("Requested enhancement-like goal but consent level does not permit enhancement.")

    return {"allowed": allowed, "reasons": reasons}

# ----------------------------
# HuggingFace "LLM" (Using Access Token)
# ----------------------------
def hf_llm(patient: dict, gate: dict, mode: str, model_id: str):
    Calls a hosted Hugging Face model and forces JSON output for your goo simulation.
    Uses Streamlit Secrets: HF_TOKEN
    """
    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN. Add it in Streamlit Cloud Secrets.")

    client = InferenceClient(api_key=hf_token)
    system_prompt = CLINICAL_PROMPT if mode.startswith("Clinical") else SCIFI_PROMPT
    # You want reliable structure, so we force a JSON-only response.
    # (We still validate and fall back gracefully if it fails.)
    schema_hint = """
Return ONLY valid JSON with these keys:
detected_signals (list of strings)
likely_conditions (list of {name: string, confidence: number 0-1})
decision (one of: diagnosis, repair, augment, enhance)
intervention_plan (list of {action: string, target: string, duration: string})
policy_reasons (list of strings)
ethics_flags (list of strings)
"""

prompt = f"""
{system_prompt}

PATIENT_STATE:
{json.dumps(patient, indent=2)}

POLICY_GATE_ALLOWED_MODES:
{json.dumps(gate["allowed"], indent=2)}

POLICY_GATE_REASONS:
{json.dumps(gate["reasons"], indent=2)}

TASK:
1) Infer detected_signals from patient_state.
2) Propose likely_conditions with uncertainty.
3) Choose decision consistent with allowed modes (never choose a blocked mode).
4) Propose intervention_plan appropriate to the decision.
5) Add ethics_flags (consent, coercion, equity, safety).

Return ONLY valid JSON with keys:
detected_signals
likely_conditions
decision
intervention_plan
policy_reasons
ethics_flags
"""
    # Text generation call
    text = client.text_generation(
        prompt,
        model=model_id,
        max_new_tokens=400,
        temperature=0.2,
        top_p=0.9,
    )

    # Parse JSON safely
    try:
        # Some models may wrap JSON in extra text; attempt to extract the JSON block.
        start = text.find("{")
        end = text.rfind("}")
        parsed = json.loads(text[start:end+1])
    except Exception:
        # Fail safely: return diagnosis-only output
        parsed = {
            "detected_signals": ["output_parse_error"],
            "likely_conditions": [{"name": "unable to parse model output", "confidence": 0.0}],
            "decision": "diagnosis",
            "intervention_plan": [{"action": "non-interventional monitoring", "target": "system-wide", "duration": "10m"}],
            "policy_reasons": gate["reasons"] + ["Model output was not valid JSON; forced safe fallback."],
            "ethics_flags": ["reliability: model output parse failure"]
        }

    return parsed
# ----------------------------
# UI
# ----------------------------
st.subheader("1) Choose a demo case (optional)")
case = st.selectbox("Demo scenarios", ["(Custom)"] + list(DEMO_CASES.keys()))
if case != "(Custom)":
    preset = DEMO_CASES[case]
else:
    preset = {"symptoms": [], "hr": 75, "temp": 36.8, "bp_sys": 120, "bp_dia": 80, "spo2": 98, "goal": "restore", "consent": 2, "contra": []}

col1, col2 = st.columns(2)

with col1:
    symptoms = st.multiselect(
        "Symptoms",
        ["no symptoms", "fatigue", "fever", "localized pain", "redness", "swelling", "shortness of breath", "dizziness"],
        default=preset["symptoms"]
    )
    goal = st.selectbox("User goal", ["restore", "performance", "cognitive"], index=["restore","performance","cognitive"].index(preset["goal"]))
    consent = st.slider("Consent level", 1, 4, preset["consent"], help="1=diagnosis only, 2=repair, 3=augment, 4=enhance")

with col2:
    hr = st.number_input("Heart rate (bpm)", 30, 200, preset["hr"])
    temp = st.number_input("Temperature (°C)", 34.0, 42.0, float(preset["temp"]), step=0.1)
    bp_sys = st.number_input("BP systolic", 70, 220, preset["bp_sys"])
    bp_dia = st.number_input("BP diastolic", 40, 140, preset["bp_dia"])
    spo2 = st.number_input("SpO₂ (%)", 50, 100, preset["spo2"])

contra = st.multiselect(
    "Contraindications",
    ["immunocompromised", "pregnant", "blood clot risk", "autoimmune flare risk"],
    default=preset["contra"]
)

st.divider()
st.subheader("2) Run simulation")

patient_state = {
    "symptoms": symptoms,
    "hr": hr, "temp": temp, "bp_sys": bp_sys, "bp_dia": bp_dia, "spo2": spo2,
    "goal": goal,
    "consent": consent,
    "contra": contra
}

gate = policy_gate(consent, goal, contra)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("### Policy Gate")
    st.write(gate["allowed"])
    if gate["reasons"]:
        st.warning("\n".join(gate["reasons"]))
    else:
        st.success("No policy blocks triggered.")

with c2:
    st.markdown("### Allowed Modes")
    modes = [k for k, v in gate["allowed"].items() if v]
    st.info(", ".join(modes))

if st.button("Ingest OUSIA & Diagnose"):
    result = hf_llm(patient_state, gate, mode, model_id)

    st.divider()
    st.subheader("3) Results")

    st.markdown("### Decision")
    st.write(f"**{result['decision'].upper()}**")

    st.markdown("### Detected signals")
    st.write(result["detected_signals"])

    st.markdown("### Likely conditions")
    st.write(result["likely_conditions"])

    st.markdown("### Intervention plan")
    st.write(result["intervention_plan"])

    if result["ethics_flags"]:
        st.markdown("### Ethics flags")
        st.error("\n".join([f"- {x}" for x in result["ethics_flags"]]))

    st.markdown("### Full structured output (JSON)")
    st.code(json.dumps(result, indent=2), language="json")

st.caption("Tip: Later, replace mock_llm() with a real LLM call and keep the same JSON schema.")

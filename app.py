# OUSIA GROUP 5 IP2 CODE FOR LLM LOGIC (Final Demo Prototype LLM)

import json
import streamlit as st
from dataclasses import dataclass, asdict

st.set_page_config(page_title="Goo Diagnosis Simulator", layout="centered")

st.title("🫧 Goo Diagnosis + Adaptive Response Simulator (2040)")
st.caption("Prototype demo: ingest → detect → decide → act, with consent + policy gating.")

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
# Mock "LLM" (works without API keys)
# ----------------------------
def mock_llm(patient: dict, gate: dict) -> dict:
    """
    A deterministic placeholder that mimics what an LLM would output.
    You can later swap this for a real model call.
    """
    symptoms = set(patient["symptoms"])
    detected = []
    likely = []
    plan = []
    ethics_flags = []

    # Simple signal inference
    if patient["temp"] >= 38.0:
        detected.append("elevated temperature (possible infection/inflammation)")
    if patient["spo2"] <= 93:
        detected.append("low oxygen saturation")
    if "swelling" in symptoms or "redness" in symptoms:
        detected.append("localized inflammation markers")

    # Simple condition guesses
    if "low oxygen saturation" in " ".join(detected):
        likely.append({"name": "respiratory compromise (non-specific)", "confidence": 0.62})
    if "elevated temperature" in " ".join(detected):
        likely.append({"name": "acute inflammatory response", "confidence": 0.66})
    if "localized inflammation markers" in detected:
        likely.append({"name": "minor tissue injury", "confidence": 0.74})

    # Decide mode based on gate
    if gate["allowed"]["repair"] and (patient["temp"] >= 38.0 or "swelling" in symptoms or patient["spo2"] <= 93):
        decision = "repair"
    elif gate["allowed"]["augment"] and patient["goal"] == "performance":
        decision = "augment"
    elif gate["allowed"]["enhance"] and patient["goal"] in ["cognitive", "performance"]:
        decision = "enhance"
    else:
        decision = "diagnosis"

    # Build plan
    if decision == "repair":
        plan = [
            {"action": "targeted anti-inflammatory release", "target": "affected tissue", "duration": "2h"},
            {"action": "micro-scaffold support", "target": "injury site", "duration": "6h"}
        ]
    elif decision == "augment":
        plan = [{"action": "temporary oxygen delivery optimization", "target": "muscle tissue", "duration": "45m"}]
        ethics_flags.append("augmentation: monitor coercion/pressure risks in competitive settings")
    elif decision == "enhance":
        plan = [{"action": "temporary neural metabolic optimization", "target": "central nervous system", "duration": "30m"}]
        ethics_flags.append("enhancement: equity/access concern + potential social coercion")
    else:
        plan = [{"action": "non-interventional monitoring", "target": "system-wide", "duration": "10m"}]

    if patient["consent"] < 3 and patient["goal"] in ["performance", "cognitive"]:
        ethics_flags.append("insufficient consent for augmentation/enhancement request")

    return {
        "detected_signals": detected or ["baseline within expected range"],
        "likely_conditions": likely or [{"name": "no abnormality detected", "confidence": 0.55}],
        "decision": decision,
        "intervention_plan": plan,
        "policy_reasons": gate["reasons"],
        "ethics_flags": ethics_flags
    }

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

if st.button("🧪 Ingest goo & run diagnosis"):
    result = mock_llm(patient_state, gate)

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

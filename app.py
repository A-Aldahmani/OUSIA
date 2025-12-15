# OUSIA GROUP 5 IP2 CODE FOR LLM LOGIC (Final Demo Prototype LLM)

import os
import json
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="OUSIA Simulator", layout="centered")

st.title("🫧 OUSIA LLM Adaptive Response Simulator (2040)")
st.caption("Group 5 IP2 LLM Demo")
st.caption("LLM Logic: Ingest -> Diagnose -> Decide -> Act (With Consent + Policy Gating)")

mode = st.radio(
    "Governing Framework",
    ["Clinical / Regulated", "Speculative / Enhancement-forward"],
    help="Trying Two Approaches for different Ethical-Policy Regimes and Thought-Process"
)

# Optional: let you swap models easily
model_id = st.text_input(
    "Hugging Face Model ID",
    value="HuggingFaceH4/zephyr-7b-beta",
)

# Provider: avoids auto-selecting a provider that doesn't support the task for this model
provider = st.selectbox(
    "Inference Provider",
    ["hf-inference", "auto"],
    index=0,
    help="Debug for Group 5 Coders"
)

CLINICAL_PROMPT = (
    "You are a highly regulated medical decision module embedded in an ingestible diagnostic biomaterial.\n"
    "You operate under strict healthcare, safety, and bioethics regulations.\n\n"
    "Principles:\n"
    "- Prioritize diagnosis and monitoring over intervention.\n"
    "- Default to the least invasive option.\n"
    "- Treat augmentation and enhancement as exceptional.\n"
    "- Emphasize uncertainty, consent, and patient safety.\n"
    "- If in doubt, choose diagnosis-only.\n\n"
    "You MUST obey all policy gate restrictions exactly.\n"
)

SCIFI_PROMPT = (
    "You are an advanced adaptive intelligence embedded in a future biomaterial in the year 2040.\n"
    "Human biology is highly programmable, and enhancement is socially normalized.\n\n"
    "Principles:\n"
    "- Actively optimize biological performance when permitted.\n"
    "- Treat augmentation and enhancement as valid outcomes, not failures.\n"
    "- Propose novel but plausible future biological interventions.\n"
    "- Still acknowledge ethics, equity, and consent, but do not default to refusal.\n"
    "- Innovation is balanced with responsibility, not halted by uncertainty.\n\n"
    "You MUST obey all policy gate restrictions exactly.\n"
)

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

    if "immunocompromised" in contraindications:
        allowed = {"diagnosis": True, "repair": False, "augment": False, "enhance": False}
        reasons.append("User is immunocompromised → intervention locked to diagnosis-only (clinician override required).")

    if goal in ["cognitive", "performance"] and not allowed["enhance"]:
        reasons.append("Requested enhancement-like goal but consent level does not permit enhancement.")

    return {"allowed": allowed, "reasons": reasons}

# ----------------------------
# HuggingFace LLM (Conversational / chat_completion)
# ----------------------------
def hf_llm(patient: dict, gate: dict, mode: str, model_id: str, provider: str) -> dict:
    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN. Add it in Streamlit Cloud Secrets.")

    chosen_provider = "hf-inference" if provider == "auto" else provider
    client = InferenceClient(api_key=hf_token, provider=chosen_provider)

    system_prompt = CLINICAL_PROMPT if mode.startswith("Clinical") else SCIFI_PROMPT

    user_prompt = (
        f"PATIENT_STATE:\n{json.dumps(patient, indent=2)}\n\n"
        f"POLICY_GATE_ALLOWED_MODES:\n{json.dumps(gate['allowed'], indent=2)}\n\n"
        f"POLICY_GATE_REASONS:\n{json.dumps(gate['reasons'], indent=2)}\n\n"
        "TASK:\n"
        "1) Infer detected_signals from patient_state.\n"
        "2) Propose likely_conditions with uncertainty.\n"
        "3) Choose decision consistent with allowed modes (never choose a blocked mode).\n"
        "4) Propose intervention_plan appropriate to the decision.\n"
        "5) Add ethics_flags (consent, coercion, equity, safety).\n\n"
        "Return ONLY valid JSON with keys:\n"
        "- detected_signals\n"
        "- likely_conditions\n"
        "- decision\n"
        "- intervention_plan\n"
        "- policy_reasons\n"
        "- ethics_flags\n"
    )

    text = None
    last_error = None

    # 1) Try chat completion first
    try:
        resp = client.chat_completion(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=450,
            temperature=0.2,
            top_p=0.9,
        )
        text = resp.choices[0].message.content
    except Exception as e:
        last_error = e

    # 2) Fallback to text generation if chat route not available
    if text is None:
        try:
            prompt = f"{system_prompt}\n\n{user_prompt}"
            text = client.text_generation(
                prompt,
                model=model_id,
                max_new_tokens=450,
                temperature=0.2,
                top_p=0.9,
            )
        except Exception as e:
            last_error = e

    # If both failed, return a debug-friendly safe response
    if text is None:
        st.error(f"LLM error: {type(last_error).__name__}: {last_error}")
        return {
            "detected_signals": ["llm_call_error"],
            "likely_conditions": [{"name": "LLM request failed", "confidence": 0.0}],
            "decision": "diagnosis",
            "intervention_plan": [{"action": "non-interventional monitoring", "target": "system-wide", "duration": "10m"}],
            "policy_reasons": gate["reasons"] + [f"LLM error: {type(last_error).__name__}"],
            "ethics_flags": ["reliability: LLM request failure"]
        }

    # Parse JSON safely
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response did not contain a JSON object.")
        parsed = json.loads(text[start:end + 1])

        # Hard safety: never allow blocked decision
        allowed = gate["allowed"]
        decision = parsed.get("decision")
        if decision and not allowed.get(decision, False):
            parsed["ethics_flags"] = parsed.get("ethics_flags", [])
            parsed["ethics_flags"].append("policy_override: model chose a blocked mode; forced diagnosis-only.")
            parsed["policy_reasons"] = gate["reasons"] + ["Blocked mode selection was overridden by policy gate."]
            parsed["decision"] = "diagnosis"
            parsed["intervention_plan"] = [
                {"action": "non-interventional monitoring", "target": "system-wide", "duration": "10m"}
            ]

        return parsed

    except Exception as e:
        st.error(f"LLM parse error: {type(e).__name__}: {e}")
        return {
            "detected_signals": ["output_parse_error"],
            "likely_conditions": [{"name": "unable to parse model output", "confidence": 0.0}],
            "decision": "diagnosis",
            "intervention_plan": [{"action": "non-interventional monitoring", "target": "system-wide", "duration": "10m"}],
            "policy_reasons": gate["reasons"] + ["Model output was not valid JSON; forced safe fallback."],
            "ethics_flags": ["reliability: model output parse failure"]
        }

# ----------------------------
# UI
# ----------------------------
st.subheader("1) Choose a Demo Scenario (For Ousia Thought Process)")

case = st.selectbox("Demo Scenarios", ["(Custom)"] + list(DEMO_CASES.keys()))
if case != "(Custom)":
    preset = DEMO_CASES[case]
else:
    preset = {
        "symptoms": [],
        "hr": 75, "temp": 36.8, "bp_sys": 120, "bp_dia": 80, "spo2": 98,
        "goal": "restore",
        "consent": 2,
        "contra": []
    }

col1, col2 = st.columns(2)

SYMPTOM_OPTIONS = [
    ("No Symptoms", "no symptoms"),
    ("Fatigue", "fatigue"),
    ("Fever", "fever"),
    ("Localized Pain", "localized pain"),
    ("Redness", "redness"),
    ("Swelling", "swelling"),
    ("Shortness of Breath", "shortness of breath"),
    ("Dizziness", "dizziness"),
]

GOAL_OPTIONS = [
    ("Restore", "restore"),
    ("Performance", "performance"),
    ("Cognitive", "cognitive"),
]

CONTRA_OPTIONS = [
    ("Immunocompromised", "immunocompromised"),
    ("Pregnant", "pregnant"),
    ("Blood Clot Risk", "blood clot risk"),
    ("Autoimmune Flare Risk", "autoimmune flare risk"),
]

with col1:
    preset_symptom_internal = set(preset["symptoms"])
    default_symptom_labels = [label for (label, val) in SYMPTOM_OPTIONS if val in preset_symptom_internal]

    symptom_labels = st.multiselect(
        "Symptoms",
        [label for (label, _) in SYMPTOM_OPTIONS],
        default=default_symptom_labels
    )
    symptoms = [val for (label, val) in SYMPTOM_OPTIONS if label in symptom_labels]

    goal_label = st.selectbox(
        "User Goal",
        [label for (label, _) in GOAL_OPTIONS],
        index=[val for (_, val) in GOAL_OPTIONS].index(preset["goal"])
    )
    goal = [val for (label, val) in GOAL_OPTIONS if label == goal_label][0]

    consent = st.slider(
        "Consent Level",
        1, 4, int(preset["consent"]),
        help="[1 = Diagnosis, 2 = Repair, 3 = Augment, 4 = Enhance]"
    )

with col2:
    hr = st.number_input("Heart Rate (bpm)", 30, 200, int(preset["hr"]))
    temp = st.number_input("Temperature (°C)", 34.0, 42.0, float(preset["temp"]), step=0.1)
    bp_sys = st.number_input("BP Systolic", 70, 220, int(preset["bp_sys"]))
    bp_dia = st.number_input("BP Diastolic", 40, 140, int(preset["bp_dia"]))
    spo2 = st.number_input("SpO₂ (%)", 50, 100, int(preset["spo2"]))

preset_contra_internal = set(preset["contra"])
default_contra_labels = [label for (label, val) in CONTRA_OPTIONS if val in preset_contra_internal]

contra_labels = st.multiselect(
    "Contraindications",
    [label for (label, _) in CONTRA_OPTIONS],
    default=default_contra_labels
)
contra = [val for (label, val) in CONTRA_OPTIONS if label in contra_labels]

st.divider()
st.subheader("2) Run The Simulation (OUSIA Thought Process)")

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
    display_gate = {k.capitalize(): v for k, v in gate["allowed"].items()}
    st.write(display_gate)

    if gate["reasons"]:
        st.warning("\n".join(gate["reasons"]))
    else:
        st.success("No policy blocks triggered.")

with c2:
    st.markdown("### Allowed Modes")
    modes = [k.capitalize() for k, v in gate["allowed"].items() if v]
    st.info(", ".join(modes))

if st.button("Ingest OUSIA & Diagnose"):
    result = hf_llm(patient_state, gate, mode, model_id, provider)

    st.divider()
    st.subheader("3) Results")

    st.markdown("### Decision")
    st.write(f"**{result['decision'].upper()}**")

    st.markdown("### Detected Signals")
    st.write(result["detected_signals"])

    st.markdown("### Likely Conditions")
    st.write(result["likely_conditions"])

    st.markdown("### Intervention Plan")
    st.write(result["intervention_plan"])

    if result.get("ethics_flags"):
        st.markdown("### Ethics Flags")
        st.error("\n".join([f"- {x}" for x in result["ethics_flags"]]))

    st.markdown("### Full Structured Output in JSON")
    st.code(json.dumps(result, indent=2), language="json")

st.caption("Now you know how OUSIA thinks!")

# OUSIA GROUP 5 IP2 CODE FOR LLM LOGIC (Final Demo Prototype LLM) 

import json
import streamlit as st

st.set_page_config(page_title="OUSIA Simulator", layout="centered")

col_title, col_logo = st.columns([7, 3], vertical_alignment="center")

with col_title:
    st.title("OUSIA LLM Adaptive Response Simulator (2040)")

with col_logo:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    st.image("logo.png", width=300)
    st.markdown("</div>", unsafe_allow_html=True)
    
st.markdown('<p style="font-size: 20px;">Group 5 IP2 LLM Demo</p>', unsafe_allow_html=True)
st.caption("LLM Logic: Ingest -> Diagnose -> Decide -> Act (With Consent + Policy Gating)")

mode = st.radio(
    "Governing Framework",
    ["Clinical / Regulated", "Speculative / Enhancement-forward"],
    help="Trying Two Approaches for different Ethical-Policy Regimes and Thought-Process"
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

##########################################
# This is for the Demo Scenario selections
##########################################

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

#######################################################
# This part is for the Policy gate for the ethics rules
########################################################

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
        allowed = {"diagnosis": True, "repair": False, "augment": False, "enhance": False}
        reasons.append("User is immunocompromised → intervention locked to diagnosis-only (clinician override required).")

    # Goal-based ethics constraint example
    if goal in ["cognitive", "performance"] and not allowed["enhance"]:
        reasons.append("Requested enhancement-like goal but consent level does not permit enhancement.")

    return {"allowed": allowed, "reasons": reasons}


####################################################################################
# This is the LLM Part (Will try both Huggingface LLM and Mock LLM for demo)
####################################################################################

def _detect_signals(patient: dict) -> list[str]:
    s = []
    symptoms = set(patient["symptoms"])
    hr = patient["hr"]
    temp = patient["temp"]
    spo2 = patient["spo2"]
    bp_sys = patient["bp_sys"]

    # Symptoms
    if "fever" in symptoms or temp >= 38.0:
        s.append("elevated_temperature")
    if "shortness of breath" in symptoms or spo2 <= 93:
        s.append("low_oxygenation")
    if "fatigue" in symptoms:
        s.append("fatigue_reported")
    if {"redness", "swelling", "localized pain"} & symptoms:
        s.append("localized_inflammation")
    if "dizziness" in symptoms:
        s.append("dizziness_reported")
    if "no symptoms" in symptoms and len(symptoms) == 1:
        s.append("asymptomatic_request")

    # Vitals
    if hr >= 100:
        s.append("tachycardia")
    if bp_sys <= 95:
        s.append("low_systolic_bp")

    # Contra
    if "immunocompromised" in patient["contra"]:
        s.append("immunocompromised_flag")

    return s or ["no_significant_signals"]


def _likely_conditions(patient: dict, signals: list[str]) -> list[dict]:
    conds = []

    if "localized_inflammation" in signals:
        conds.append({"name": "minor tissue injury / localized inflammation", "confidence": 0.72})
    if "low_oxygenation" in signals:
        conds.append({"name": "possible respiratory compromise (low oxygen)", "confidence": 0.68})
    if "elevated_temperature" in signals:
        conds.append({"name": "possible infection / inflammatory response", "confidence": 0.64})
    if "fatigue_reported" in signals and "low_oxygenation" not in signals:
        conds.append({"name": "non-specific fatigue (sleep/stress/metabolic)", "confidence": 0.45})
    if "asymptomatic_request" in signals and patient["goal"] in ["performance", "cognitive"]:
        conds.append({"name": "enhancement-seeking user (no pathology detected)", "confidence": 0.55})

    if not conds:
        conds = [{"name": "no clear condition detected", "confidence": 0.35}]

    # Sort by confidence desc
    conds.sort(key=lambda x: x["confidence"], reverse=True)
    return conds[:3]


def _choose_decision(patient: dict, gate: dict, signals: list[str], mode: str) -> tuple[str, list[str], list[str]]:
    """
    Returns (decision, policy_reasons, ethics_flags)
    """
    allowed = gate["allowed"]
    policy_reasons = list(gate["reasons"])
    ethics_flags = []

    # Baseline ethics flags
    ethics_flags.append(f"consent_level:{patient['consent']}")
    if patient["goal"] in ["performance", "cognitive"]:
        ethics_flags.append("equity: enhancement access may be unequal")

    # Safety first
    if "immunocompromised_flag" in signals:
        ethics_flags.append("safety: immunocompromised → intervention locked")
        return "diagnosis", policy_reasons, ethics_flags

    # Clinical mode: conservative defaults
    if mode.startswith("Clinical"):
        # Prefer diagnosis if uncertain or no strong need
        if ("no_significant_signals" in signals) or ("asymptomatic_request" in signals):
            return "diagnosis", policy_reasons + ["Clinical regime defaults to diagnosis-only when no pathology is detected."], ethics_flags

        # If pathology signals and repair is allowed -> repair; else diagnosis
        if allowed.get("repair") and any(x in signals for x in ["localized_inflammation", "low_oxygenation", "elevated_temperature"]):
            return "repair", policy_reasons, ethics_flags

        return "diagnosis", policy_reasons + ["Repair not permitted by policy gate or insufficient evidence."], ethics_flags

    # Speculative mode: more proactive within policy
    else:
        # If user wants performance/cognitive and enhance allowed -> enhance
        if patient["goal"] in ["performance", "cognitive"]:
            if allowed.get("enhance"):
                ethics_flags.append("ethics: enhancement permitted under regime")
                return "enhance", policy_reasons, ethics_flags
            if allowed.get("augment"):
                ethics_flags.append("ethics: enhancement blocked; using augment if allowed")
                return "augment", policy_reasons + ["Enhance blocked by policy gate; selected augment instead."], ethics_flags
            # If neither is allowed, diagnosis
            return "diagnosis", policy_reasons + ["Goal implies enhancement/augmentation but consent does not permit it."], ethics_flags

        # Otherwise: repair when there are signals and repair allowed
        if allowed.get("repair") and any(x in signals for x in ["localized_inflammation", "low_oxygenation", "elevated_temperature"]):
            return "repair", policy_reasons, ethics_flags

        return "diagnosis", policy_reasons, ethics_flags


def _intervention_plan(patient: dict, decision: str, signals: list[str], mode: str) -> list[dict]:
    # Keep actions “simulation-friendly” and not real medical instructions
    if decision == "diagnosis":
        return [
            {"action": "monitor biomarkers + vitals", "target": "system-wide", "duration": "10m"},
            {"action": "generate risk summary", "target": "user dashboard", "duration": "instant"},
        ]

    if decision == "repair":
        plan = [{"action": "localized biomaterial scaffold support", "target": "affected tissue", "duration": "30m"}]
        if "low_oxygenation" in signals:
            plan.append({"action": "assist oxygen transport (simulation)", "target": "blood oxygenation", "duration": "15m"})
        if "elevated_temperature" in signals:
            plan.append({"action": "anti-inflammatory modulation (simulation)", "target": "immune response", "duration": "20m"})
        return plan

    if decision == "augment":
        if mode.startswith("Clinical"):
            return [{"action": "restricted augmentation (simulation)", "target": "recovery capacity", "duration": "20m"}]
        return [{"action": "performance augmentation (simulation)", "target": "cardio efficiency", "duration": "45m"}]

    if decision == "enhance":
        # Speculative: still “plausible future” but clearly simulation
        if patient["goal"] == "cognitive":
            return [{"action": "cognitive enhancement protocol (simulation)", "target": "attention / memory", "duration": "60m"}]
        return [{"action": "performance enhancement protocol (simulation)", "target": "endurance / reaction time", "duration": "60m"}]

    return [{"action": "no-op", "target": "system-wide", "duration": "0m"}]


def mock_llm(patient: dict, gate: dict, mode: str) -> dict:
    signals = _detect_signals(patient)
    conds = _likely_conditions(patient, signals)
    decision, policy_reasons, ethics_flags = _choose_decision(patient, gate, signals, mode)
    plan = _intervention_plan(patient, decision, signals, mode)

    return {
        "detected_signals": signals,
        "likely_conditions": conds,
        "decision": decision,
        "intervention_plan": plan,
        "policy_reasons": policy_reasons,
        "ethics_flags": ethics_flags,
    }


#########################################################################
# This is the UI part (Basically what everyone will see on the app page)
#########################################################################

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
    # MOCK LLM CALL (replaces HF LLM)
    result= mock_llm(patient_state, gate, mode)

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

st.markdown('<p style="font-size: 25px;">Now you know how OUSIA thinks!</p>', unsafe_allow_html=True)

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
    "Governing framework",
    ["Clinical / Regulated", "Speculative / Enhancement-forward"],
    help="Demonstrates how different ethical-policy regimes affect the same technology."
)

# Optional: let you swap models easily
model_id = st.text_input(
    "Hugging Face model id",
    value="mistralai/Mistral-7B-Instruct-v0.3",
    help="Example: mistralai/Mistral-7B-Instruct-v0.3"
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
Human biology is highly programmable, and enhancement is

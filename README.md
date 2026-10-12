# 💊 MedAgent: Handwritten Prescription Detection & Clinical Decision Support

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![TrOCR](https://img.shields.io/badge/Model-Microsoft%20TrOCR-orange.svg)](https://huggingface.co/microsoft/trocr-small-handwritten)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%201.47-red.svg)](https://streamlit.io)
[![Clinical Safety](https://img.shields.io/badge/Safety-LASA%20%7C%20DDI%20%7C%20MaxDose-green.svg)](#clinical-decision-support--guardrails)

**MedAgent** is an AI-assisted prescription extraction and clinical decision-support prototype. It scans handwritten doctor prescriptions, extracts line-level text using Vision-Transformer OCR (`microsoft/trocr-small-handwritten`), decodes quirky medical shorthand and cursive doctor handwriting, validates medications against a standardized drug formulary, and flags clinical safety hazards for human-in-the-loop (HITL) pharmacist verification.

---

## 🌟 Key Capabilities

1. **Prescription Pad Image Preprocessing & Line Slicing**
   - Auto-contrast enhancement and adaptive thresholding to clarify faint ballpoint pen ink.
   - Horizontal projection profile line segmentation isolating each medication order strip.
   - Ink stroke sharpening and binarization.

2. **Quirky Doctor Handwriting & Shorthand Decoding**
   - Transformer-based handwriting OCR (`TrOCR`) fine-tuned on cursive scripts.
   - Ligature and cursive post-corrections (e.g., resolving `l-0-l` -> `1-0-1`, `SOOmg` -> `500mg`).
   - Medical abbreviation expander:
     - Timing: `1-0-1`, `1-1-1`, `0-0-1`, `OD`, `BD`, `TID`, `QID`, `HS`, `SOS`, `STAT`.
     - Durations: `x 5d`, `x 3/7` (3 days), `x 2/52` (2 weeks), `x 1/12` (1 month).
     - Meal instructions: `ac` (before meals), `pc` (after meals), `bbf` (before breakfast), `empty stomach`.
     - Dosage forms: `Tab`, `Cap`, `Syr`, `Inj`, `Drops`, `Oint`.

3. **Formulary Validation & Fuzzy Drug Resolution**
   - Curated reference database of common essential medications across primary care, cardiology, antimicrobial therapy, and endocrinology.
   - Dual-layer fuzzy string matching: Normalized Levenshtein ratio (`rapidfuzz`) + American Soundex phonetic encoding.
   - Gracefully resolves misspellings and cursive OCR distortions (e.g. *"Paracefamol"* -> *Paracetamol*, *"Metformn"* -> *Metformin*, *"Dolo 650"* -> *Paracetamol / Dolo*).

4. **Clinical Decision Support & Safety Guardrails**
   - **Look-Alike Sound-Alike (LASA) Drug Warnings**: Detects high-risk confused pairs (e.g. `celeCOXIB` vs `citALOPram`, `cloniDINE` vs `clonaZEPAM`, `hydrALAZINE` vs `hydrOXYzine`) with Tall Man lettering.
   - **Drug-Drug Interactions (DDI)**: Flags life-threatening combinations such as `Warfarin + Ibuprofen` (severe GI hemorrhage risk) and `Lisinopril + Spironolactone` (hyperkalemia risk).
   - **Maximum 24h Daily Dose Limits**: Checks cumulative daily intake against therapeutic toxicity ceilings (e.g. Paracetamol > 4000 mg/day).
   - **OCR Uncertainty Flags**: Highlights low-confidence words (< 75%) with visual warning badges to prompt human pharmacist confirmation.

5. **Interactive Pharmacist Review & Digital Dispensing**
   - Side-by-side visual line inspection showing original prescription crops alongside extracted data.
   - Inline editable fields allowing the pharmacist to approve, modify, or reject items.
   - One-click export to standardized **FHIR-compliant JSON** and clinical dispensing summaries.
   - Preloaded **5 Benchmark Clinical Sample Cases** ready for immediate testing.

---

## 🏗️ Architecture

```
                    ┌────────────────────────┐
                    │  Handwritten Scan Pad   │
                    │   (Upload or Benchmark)│
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  Image Preprocessor    │
                    │  • CLAHE / Contrast    │
                    │  • Line Slicing/Crop   │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  TrOCR Vision Model    │
                    │  • Confidence Scoring  │
                    │  • Handwriting Fixes   │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  Clinical NLP Parser   │
                    │  • Medicine / Strength │
                    │  • Frequency (1-0-1)   │
                    │  • Duration (x 5/7)    │
                    │  • Route & Meal Timing │
                    └───────────┬────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
    ┌──────────────────────┐          ┌──────────────────────┐
    │ Drug Formulary DB    │          │ Safety Guardrails    │
    │ • Fuzzy Matching     │          │ • DDI Detector       │
    │ • Soundex Phonetics  │          │ • LASA Tall Man      │
    │ • RxNorm / FDA Names │          │ • Max Daily Dose     │
    └──────────┬───────────┘          └──────────┬───────────┘
               │                                 │
               └────────────────┬────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Streamlit Clinical UI  │
                    │ • Pharmacist Sign-Off  │
                    │ • FHIR / JSON Export   │
                    └────────────────────────┘
```

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.13)
- PyTorch (CPU or CUDA)

### Installation
Clone or navigate to the repository directory:
```bash
cd C:\Older\OneDrive\Desktop\Medsure
py -3.13 -m pip install -r requirements.txt
```

### Running the Test Suite
MedAgent includes 14 unit and end-to-end integration tests:
```bash
py -3.13 -m unittest discover -s tests -p "test_*.py"
```

### Launching the Web Application
Start the Streamlit dashboard:
```bash
py -3.13 -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Benchmark Test Scenarios Included

MedAgent comes preloaded with 5 diverse benchmark cases accessible from the sidebar:
1. **Sample 1: Acute Infection & Fever**: Amoxicillin 500mg TID + Dolo 650 (1-0-1) + Pantocid 40 (ac).
2. **Sample 2: Chronic Care Regimen**: Metformin 500mg (pc) + Telmisartan 40mg + Atorvastatin 10mg (hs).
3. **Sample 3: Critical DDI Alert**: Warfarin 5mg + Brufen 400mg (Severe bleeding hemorrhage alert).
4. **Sample 4: High-Risk LASA & Max Dose Warning**: Celebrex 200mg (LASA with Celexa) + Paracetamol 4000mg limit test.
5. **Sample 5: Pediatric & Latin Shorthand**: Augmentin 625 BD x 5/7 (fraction duration) + Omez 20 OD bbf.

---

## 🛡️ Clinical Disclaimer
MedAgent is an AI decision-support prototype. It is designed to assist, not replace, certified healthcare practitioners and licensed pharmacists. All medication orders must undergo qualified clinical review before dispensing.

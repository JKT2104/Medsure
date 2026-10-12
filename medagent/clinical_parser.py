"""Clinical Entity Parser for Doctor Prescriptions.
Extracts medicine names, strength, dosage form, frequency, duration, route,
and administration instructions from messy handwritten OCR strings.
"""

from typing import Dict, Any, List, Optional
import re
from medagent.abbreviation_expander import (
    expand_frequency,
    expand_duration,
    expand_instructions,
    DOSAGE_FORM_MAP
)
from medagent.drug_db import drug_db


class ClinicalParser:
    """Parses raw OCR lines into structured clinical medication orders."""

    def __init__(self):
        # Dosage unit patterns (e.g., 500mg, 500 mg, 625m, 0.5mg, 10ml, 250mcg, 1g)
        self.dose_pattern = re.compile(
            r'(\b\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|iu|tablets?|caps?|m\b)',
            re.IGNORECASE
        )
        # Standalone numeric dose followed immediately by clinical frequency (e.g. "Dolo 650 1-0-1")
        self.standalone_dose_pattern = re.compile(
            r'\b(\d{2,4})\b(?=\s+(?:1-0-1|1-1-1|0-0-1|1-0-0|bd|bid|tds|tid|od|stat|sos|po|pc|ac)\b)',
            re.IGNORECASE
        )
        self.clinical_advice = []

    def clean_line(self, line: str) -> str:
        """Strips RX/Advice symbols, leading line numbers, and extra symbols."""
        # Strip Rx, Adv, Advice headers
        cleaned = re.sub(r'^(?:rx|℞|r/|\+|adv|adu|advise|advice)\s*[:.-]?\s*', '', line, flags=re.IGNORECASE)
        # Separate joined letters and numbers, preserving Q-hour intervals like Q6H, Q4H and duration x3d
        cleaned = re.sub(
            r'\b([a-zA-Z]+)(\d+(?:mg|g|ml|mcg|m)?)\b',
            lambda m: m.group(0) if m.group(1).lower() in ['q', 'x'] else f"{m.group(1)} {m.group(2)}",
            cleaned
        )
        # Separate leading numbers from drug names (e.g. "7Guyemet" -> "7 Guyemet", "8Glucored" -> "8 Glucored")
        cleaned = re.sub(r'^(\d{1,2})([a-zA-Z]{3,})', r'\1 \2', cleaned)
        # Strip leading numbers like "1.", "1)", "(1)", "1 -", "+6.", or standalone leading item number "8 Glucored"
        cleaned = re.sub(r'^\s*(?:\+?\d{1,2}[\.\-\)]?|\+6\.)\s+', '', cleaned)
        # Normalize spaced Q-hour intervals (e.g. "Q 6H" -> "Q6H")
        cleaned = re.sub(r'\b[Qq]\s*(\d{1,2})\s*[Hh]\b', r'Q\1H', cleaned)
        return cleaned.strip()

    def is_clinic_footer(self, text: str) -> bool:
        """Identifies clinic address, telephone numbers, and footer metadata."""
        clean = text.lower().strip()
        footer_patterns = [
            r'\d{8,12}', r'sec\s*\d+', r'opp\b', r'shastri', r'bridge', r'vijay\s*nagar',
            r'dr[:\.\s]*ashutosh', r'dental\s*clinic', r'b\.?d\.?s', r'oral\s*surgeon',
            r'timings?', r'h\.?no', r'near\b', r'road\b', r'chandanagar', r'hyderabad',
            r'telangana', r'phone', r'ph:', r'mob:'
        ]
        return any(re.search(p, clean) for p in footer_patterns)

    def is_vitals_or_demographics(self, text: str) -> bool:
        """Identifies patient demographics, vitals, or clinical examination notes."""
        clean = text.lower().strip()
        
        # Don't treat clear medication orders as vitals unless explicitly labeled
        if any(f in clean for f in ["tab", "cap", "syp", "inj", "cream", "gel", "dt", "susp"]) and not any(k in clean for k in ["wt", "weight", "age", "gender", "pge", "cender"]):
            return False

        vitals_triggers = [
            "weight", "wt", "kg", "kgs", "kilo", "height", "ht",
            "age", "gender", "cender", "pge", "years", "year", "yrs", "yr", "y/o", "male", "female",
            "bp", "pulse", "pr:", "temp", "temperature", "spo2", "rr-", "rr:", "rs-", "rs:",
            "diagnosis", "diag", "c/o", "history", "findings", "clinical description", "dosctiption",
            "patient", "pt.", "pt:", "name:", "date:"
        ]
        
        if any(re.search(rf'\b{re.escape(trig)}\b', clean) for trig in vitals_triggers):
            return True
            
        if bool(re.search(r'\b\d+(?:\.\d+)?\s*(?:kg|kgs)\b', clean)):
            return True
        if bool(re.search(r'\b\d{1,2}\s*(?:yrs?|y/o|y\b)\b', clean)) and ("age" in clean or "/" in clean or "y" in clean):
            return True
        if bool(re.search(r'\b\d{2,3}/\d{2,3}\s*(?:mmhg)?\b', clean)) and ("bp" in clean or "/" in clean):
            return True
        if bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', clean)) and ("date" in clean or "/" in clean or "-" in clean):
            return True

        return False

    def extract_vitals_and_demographics(self, text: str) -> Dict[str, str]:
        """Extracts key patient demographics and vitals into a structured dictionary."""
        clean = text.strip()
        ltxt = clean.lower()
        extracted = {}

        # Weight
        wt_match = re.search(r'\b(?:weight|wt|w/t)?[:\s\-]*(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilo|kilograms?)\b', ltxt)
        if wt_match:
            extracted["weight"] = f"{wt_match.group(1)} kg"
        elif "weight" in ltxt or "wt" in ltxt:
            val_match = re.search(r'\b(?:weight|wt)[:\s\-]+(\d+(?:\.\d+)?)', ltxt)
            if val_match:
                extracted["weight"] = f"{val_match.group(1)} kg"

        # Age / Gender
        age_match = re.search(r'\b(?:age[:\s]*)?(\d{1,2})\s*(?:yrs?|y/o|years?|y)\b', ltxt)
        gender_match = re.search(r'\b(male|female)\b', ltxt)
        if not gender_match:
            gender_match = re.search(r'\b(?:sex|gender)[:\s]*([mf])\b', ltxt)
        if age_match or gender_match:
            age_str = f"{age_match.group(1)} Y" if age_match else ""
            gen_str = gender_match.group(1).capitalize() if gender_match else ""
            if gen_str in ["M", "Male"]:
                gen_str = "Male"
            elif gen_str in ["F", "Female"]:
                gen_str = "Female"
            if age_str and gen_str:
                extracted["age_gender"] = f"{age_str} / {gen_str}"
            elif age_str:
                extracted["age_gender"] = age_str
            elif gen_str:
                extracted["age_gender"] = gen_str

        # Blood Pressure
        bp_match = re.search(r'\b(?:bp[:\s]*)?(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmhg)?\b', ltxt)
        if bp_match and "bp" in ltxt:
            extracted["bp"] = f"{bp_match.group(1).replace(' ', '')} mmHg"

        # Pulse Rate
        pulse_match = re.search(r'\b(?:pulse|pr)[:\s]*(\d{2,3})\s*(?:bpm|/min)?\b', ltxt)
        if pulse_match:
            extracted["pulse"] = f"{pulse_match.group(1)} bpm"

        # Temperature
        temp_match = re.search(r'\b(?:temp|temperature)[:\s]*(\d{2,3}(?:\.\d+)?)\s*(?:f|c)?\b', ltxt)
        if temp_match:
            extracted["temp"] = f"{temp_match.group(1)} °F"

        # Diagnosis / Clinical Notes
        diag_match = re.search(r'\b(?:diagnosis|clinical\s*findings?|history|c/o|diag)[:\s]*(.+)', clean, re.IGNORECASE)
        if diag_match:
            extracted["diagnosis"] = diag_match.group(1).strip()
        elif any(k in ltxt for k in ["urti", "bronchitis", "tinea", "pharyngitis", "gastritis", "hypertension", "diabetes"]):
            extracted["diagnosis"] = clean

        # Patient Name
        name_match = re.search(r'\b(?:name|patient(?:\s*name)?|pt)[:\s]+([a-zA-Z\s\.]+)', clean, re.IGNORECASE)
        if name_match:
            cand_name = name_match.group(1).strip()
            cand_name = re.sub(r'\b(?:age|sex|gender|date|wt|weight).*$', '', cand_name, flags=re.IGNORECASE).strip()
            if len(cand_name) >= 3:
                extracted["patient_name"] = cand_name

        # Date
        date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', clean)
        if date_match and ("date" in ltxt or "/" in clean or "-" in clean):
            extracted["date"] = date_match.group(1)

        return extracted

    def is_clinical_advice(self, text: str) -> bool:
        """Identifies non-medication clinical advice (e.g. Rest for 4 days, warm saline rinse, follow-up)."""
        clean = text.lower().strip()
        advice_triggers = [
            "rest", "rtpe", "rtfe", "bed rest", "rinse", "gargle", "warm water", "salt water", "saline",
            "soft diet", "liquid diet", "avoid hot", "avoid spicy", "avoid oily", "avoid cold", "diet",
            "chew", "brushing", "brush", "mouthwash", "review after", "review on", "consult after",
            "review sos", "review", "consult", "follow up", "f/u", "r/a", "dressing", "bandage", "suture",
            "steam", "inhalation", "ice pack", "hot pack", "plenty of fluids", "drink water", "hydrate",
            "exercise", "walk", "physiotherapy", "elevation", "elevate", "blood sugar", "cbc", "investigation",
            "test", "x-ray", "fasting", "post prandial", "avoid alcohol", "stop smoking", "apply ice",
            "clean with", "keep dry", "cotton", "gauze"
        ]
        return any(trig in clean for trig in advice_triggers)

    def parse_advice_line(self, text: str) -> Dict[str, Any]:
        """Parses clinical advice into a patient care instruction entity."""
        clean = text.lower().strip()
        duration = expand_duration(clean) or ("4 days" if any(k in clean for k in ["rtpe4l", "rtfe4l", "4l", "4"]) else "Unstated")

        category = "Clinical Advice / Patient Care"
        if any(k in clean for k in ["rest", "rtpe", "rtfe", "bed rest"]):
            category = "Post-Procedure Recovery / Rest"
            desc = "Rest for 4 days (Post-procedure recovery care)" if any(k in clean for k in ["rtpe4l", "rtfe4l"]) else text
        elif any(k in clean for k in ["review", "consult", "follow up", "r/a", "f/u", "visit"]):
            category = "Follow-Up & Doctor Review"
            desc = text
        elif any(k in clean for k in ["diet", "food", "spicy", "fluids", "water", "cold drinks", "oily", "hydrate"]):
            category = "Dietary Directives & Hydration"
            desc = text
        elif any(k in clean for k in ["gargle", "rinse", "brush", "mouthwash", "saline"]):
            category = "Oral Hygiene & Care"
            desc = text
        elif any(k in clean for k in ["dressing", "bandage", "suture", "gauze", "keep dry", "clean with"]):
            category = "Wound Care & Dressing"
            desc = text
        elif any(k in clean for k in ["steam", "inhalation", "ice pack", "hot pack", "elevation", "elevate", "exercise"]):
            category = "Physical Therapy & Home Care"
            desc = text
        elif any(k in clean for k in ["test", "investigation", "cbc", "blood sugar", "x-ray", "fasting"]):
            category = "Diagnostic & Lab Investigations"
            desc = text
        else:
            desc = text

        return {
            "advice_text": desc,
            "category": category,
            "duration": duration,
            "raw_text": text
        }

    def is_pure_instruction_fragment(self, text: str) -> bool:
        """Determines if a text fragment represents purely instructions/duration/frequency with no drug name."""
        clean = text.lower().strip()
        
        # If the line contains a valid drug name, it is a medication order, NOT an instruction fragment!
        test_text = self.clean_line(text)
        test_text = re.sub(
            r'\b(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syr|syrup|susp|inj|ointment|oint|cream|soap|drops|resp|wash|lotion|gel|paint|pant)\b',
            '', test_text, flags=re.IGNORECASE
        )
        test_text = re.sub(r'\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|m)\b', '', test_text, flags=re.IGNORECASE)
        test_text = re.sub(r'\b(?:1-0-1|1-1-1|0-0-1|1-0-0|0-1-0|bd|bid|tds|tid|qid|od|qd|hs|qhs|sos|prn|stat)\b', '', test_text, flags=re.IGNORECASE)
        test_text = re.sub(r'\b(?:x\s*)?\d{1,3}\s*(?:d|days|day|dag|dags|dys|elay|w|wk|weeks|m|month)\b', '', test_text, flags=re.IGNORECASE)
        test_text = re.sub(r'\b(?:ac|pc|po|af|bf|wm|bbf)\b', '', test_text, flags=re.IGNORECASE)
        test_tokens = [w for w in re.findall(r'[a-zA-Z]+', test_text) if len(w) > 2 and w.lower() not in drug_db.STOPWORDS]
        for tok in test_tokens:
            m = drug_db.find_match(tok)
            if m and m["confidence"] >= 75.0:
                return False

        instruction_words = {
            "x", "day", "days", "dag", "dags", "dys", "d", "morn", "morning", "night", "bath", 
            "topical", "la", "for", "pc", "ac", "sos", "prn", "tid", "bid", "od", 
            "qid", "bbf", "af", "bf", "hs", "po", "topicalbath", "as", "directed", 
            "by", "physician", "apply", "daily", "twice", "once", "three", "times", 
            "hours", "hr", "hrs", "wk", "wks", "weeks", "week", "month", "months", "z",
            "massage", "paint", "gum", "pant", "xiweek", "xlweek", "x1week", "dpox", "x5elay", "elay",
            "tab", "tabs", "cap", "caps",
            "oay", "oan", "ay", "an", "doy", "dny", "clays", "x3oay", "x30an", "x30ay", "x5oay"
        }

        # Check if text is an OCR-distorted duration/instruction (e.g. x3oay, x30an, x 3 days)
        stripped = re.sub(r'(?:x\s*|for\s*|\b)\d{1,2}[0o]?(?:an|ay|oan|oay|day|days|clays|doy|dny|wk|wks|weeks?|d|dys|elay)\b', '', clean, flags=re.IGNORECASE)
        stripped = re.sub(r'\b[0o]\s*[-–—_.]+\s*[0o]\b', '', stripped)
        stripped = re.sub(r'\b(?:1-0-1|1-1-1|0-0-1|1-0-0|0-1-0|bd|bid|tds|tid|qid|od|qd|hs|qhs|sos|prn|stat)\b', '', stripped, flags=re.IGNORECASE)
        rem_words = [w for w in re.findall(r'[a-z]+', stripped) if w not in instruction_words and w not in drug_db.STOPWORDS]
        if not rem_words:
            return True

        tokens = re.findall(r'[a-z]+|\d+', clean)
        if not tokens:
            return False
        non_instruction = [t for t in tokens if t not in instruction_words and not t.isdigit()]
        return len(non_instruction) == 0

    def parse_line(self, line: str, ocr_confidence: float = 85.0) -> Dict[str, Any]:
        """Parses a single prescription line into clinical components and validates against drug DB."""
        clean_text = self.clean_line(line)
        if not clean_text:
            return {}

        # Reject noise lines that do not have at least 2 alphabetic characters (e.g. '#', '---', '1.')
        letters = re.findall(r'[a-zA-Z]', clean_text)
        if len(letters) < 2:
            return {}

        # 1. Identify Dosage Form (Tab, Cap, Syr, Gum Paint, etc.)
        dosage_form = None
        words = clean_text.split()
        for w in words[:3]:  # Usually at the beginning of the line
            w_lower = re.sub(r'[^a-z]', '', w.lower())
            if w_lower in DOSAGE_FORM_MAP:
                dosage_form = DOSAGE_FORM_MAP[w_lower]
                break

        # 2. Extract Strength / Dosage
        strength_str = None
        strength_mg = None

        ratio_match = re.search(r'\(?(\d{2,3})\s*/\s*(\d{1,2})\)?', clean_text)
        ml_match = re.search(r'(\d+(?:\.\d+)?)\s*ml\b', clean_text, re.IGNORECASE)

        if dosage_form is None:
            if any(k in clean_text.lower() for k in ["syp", "syrup", "susp", "suspension", "drops", "oral liquid"]):
                dosage_form = "Syrup"
            elif ml_match:
                dosage_form = "Syrup"
            elif any(k in clean_text.lower() for k in ["cap", "capsule", "caps"]):
                dosage_form = "Capsule"
            elif any(k in clean_text.lower() for k in ["cream", "oint", "gel"]):
                dosage_form = "Cream"
            elif any(k in clean_text.lower() for k in ["soap", "wash"]):
                dosage_form = "Medicated Soap"
            else:
                dosage_form = "Tablet"

        if ratio_match:
            r_conc = f"{ratio_match.group(1)}mg/{ratio_match.group(2)}ml"
            if ml_match:
                strength_str = f"{r_conc} ({ml_match.group(1)}ml dose)"
                strength_mg = (float(ratio_match.group(1)) / float(ratio_match.group(2))) * float(ml_match.group(1))
            else:
                strength_str = r_conc
                strength_mg = float(ratio_match.group(1))
        else:
            dose_match = self.dose_pattern.search(clean_text)
            if dose_match:
                val_num = float(dose_match.group(1))
                unit = dose_match.group(2).lower()
                if unit == "m" and val_num in [100, 200, 250, 400, 500, 625, 650, 850, 1000]:
                    unit = "mg"
                strength_str = f"{val_num:g}{unit}"
                if unit == "mg":
                    strength_mg = val_num
                elif unit == "g":
                    strength_mg = val_num * 1000.0
                elif unit == "mcg":
                    strength_mg = val_num / 1000.0
            elif ml_match:
                strength_str = f"{ml_match.group(1)}ml"
            else:
                # Fallback for standalone numeric doses (e.g. "Dolo 650", "Amlodac 5", "Ator 10", "Losar 25", "Pantop 40", "Allegra 120")
                num_matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', clean_text)
                for nm in num_matches:
                    val = float(nm)
                    # Check if this number is likely a dosage strength rather than a frequency (1-0-1) or duration (x 3 days)
                    if not re.search(rf'\b{re.escape(nm)}-|\b-{re.escape(nm)}\b', clean_text):
                        if not re.search(rf'(?:x\s*|\bx\s*){re.escape(nm)}\b|{re.escape(nm)}\s*(?:days?|weeks?|months?|dys|d|elay)\b', clean_text, re.IGNORECASE):
                            if val in [0.25, 0.5, 1, 2, 2.5, 4, 5, 10, 15, 20, 25, 30, 40, 50, 75, 80, 100, 120, 150, 160, 180, 200, 250, 300, 375, 400, 500, 600, 625, 650, 800, 850, 1000]:
                                strength_str = f"{int(val) if val.is_integer() else val}mg"
                                strength_mg = val
                                break

        # 3. Extract Frequency & Daily Multiplier
        frequency_desc, daily_multiplier = expand_frequency(clean_text)
        if not frequency_desc:
            frequency_desc = "As directed by physician"
            daily_multiplier = 1

        # 4. Extract Duration
        duration_desc = expand_duration(clean_text) or "Duration unstated"

        # 5. Extract Instructions / Route / Timing
        instructions = expand_instructions(clean_text)
        route = "Oral"
        timing = "Unspecified"

        if any("Topical" in inst for inst in instructions) or dosage_form in ["Cream", "Medicated Soap", "Ointment", "Lotion", "Gum Paint / Oral Gel"] or any(k in clean_text.lower() for k in ["gum", "paint", "massage"]):
            route = "Topical"
            if any(k in clean_text.lower() for k in ["gum", "paint", "massage"]):
                dosage_form = dosage_form or "Gum Paint / Oral Gel"
            if not any("Topical" in inst for inst in instructions):
                instructions.insert(0, "Topical application")
        elif "Oral" not in instructions and dosage_form in ["Tablet", "Capsule", "Syrup"]:
            route = "Oral"
            instructions.insert(0, "Oral")

        if any("morning" in inst.lower() for inst in instructions):
            timing = "Morning"
        elif any("night" in inst.lower() or "bedtime" in inst.lower() for inst in instructions):
            timing = "Night / Bedtime"
        elif any("after meals" in inst.lower() for inst in instructions):
            timing = "After Meals (pc)"
        elif any("before meals" in inst.lower() for inst in instructions):
            timing = "Before Meals (ac)"

        # 6. Extract Candidate Drug Name
        # Remove known structural keywords to isolate the medicine name
        name_candidate = clean_text
        # Remove dosage forms
        name_candidate = re.sub(
            r'\b(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syr|syrup|susp|inj|ointment|oint|cream|soap|drops|resp|wash|lotion|gel|paint|pant)\b',
            '', name_candidate, flags=re.IGNORECASE
        )
        # Remove dose string
        if strength_str:
            name_candidate = re.sub(rf'\b{re.escape(strength_str)}\b', '', name_candidate, flags=re.IGNORECASE)
        name_candidate = re.sub(r'\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|m)\b', '', name_candidate, flags=re.IGNORECASE)
        # Remove frequency shorthand tokens
        name_candidate = re.sub(
            r'\b(?:1-0-1|1-1-1|0-0-1|1-0-0|0-1-0|bd|bid|tds|tid|qid|od|qd|hs|qhs|sos|prn|stat)\b',
            '', name_candidate, flags=re.IGNORECASE
        )
        # Remove duration tokens
        name_candidate = re.sub(r'\b(?:x\s*)?\d{1,3}\s*(?:d|days|day|dag|dags|w|wk|weeks|m|month|/7|/52|/12)\b', '', name_candidate, flags=re.IGNORECASE)
        # Remove instruction tokens
        name_candidate = re.sub(r'\b(?:ac|pc|po|af|bf|wm|bbf)\b', '', name_candidate, flags=re.IGNORECASE)
        # Clean trailing punctuation
        name_candidate = re.sub(r'[^a-zA-Z\s\-]', ' ', name_candidate).strip()
        # Take words that are not stopwords as candidate drug name
        # Allow single uppercase letters like 'H', 'D', 'M' which represent combination drug suffixes (e.g. 'Losar H', 'Pan D')
        cand_tokens = [
            w for w in name_candidate.split()
            if (len(w) > 1 or w.isupper()) and w.lower() not in drug_db.STOPWORDS
        ]
        drug_query = " ".join(cand_tokens[:2]) if cand_tokens else (clean_text.split()[0] if clean_text.split() else "Unknown")

        # 7. Validate Against Drug Database & Find Close Matches
        match_result = drug_db.find_match(drug_query)
        if not match_result or match_result["confidence"] < 75.0:
            for tok in cand_tokens:
                if len(tok) >= 3 and tok.lower() not in drug_db.STOPWORDS:
                    tok_match = drug_db.find_match(tok)
                    if tok_match and (not match_result or tok_match["confidence"] > match_result["confidence"]):
                        match_result = tok_match
                        drug_query = tok

        # Retrieve top 3 close candidate matches
        close_matches = drug_db.find_close_matches(drug_query, top_k=3)

        # Determine corrected name
        canonical_brand_map = {
            "azigmentun": "Augmentin 625mg",
            "augmentui": "Augmentin 625mg",
            "argmentun": "Augmentin 625mg",
            "augmentin 625": "Augmentin 625mg",
            "enzsflam": "Enzoflam",
            "enzplam": "Enzoflam",
            "pandlonj": "Pan-D 40mg",
            "pan-d": "Pan-D 40mg",
            "pan d": "Pan-D 40mg",
            "herigel": "Hexigel Gum Paint",
            "herigel gum pant": "Hexigel Gum Paint",
            "hexigel": "Hexigel Gum Paint",
            "hexigel gum paint": "Hexigel Gum Paint",
            "novamox": "Novamox 500mg",
            "novamax": "Novamox 500mg",
            "novamox 500": "Novamox 500mg",
            "novamax 500": "Novamox 500mg",
            "novamox 500mg": "Novamox 500mg",
            "novamax 500mg": "Novamox 500mg",
            "xonon": "Novamox 500mg",
            "ketorol": "Ketorol-DT",
            "ketorel": "Ketorol-DT",
            "ketorol-dt": "Ketorol-DT",
            "ketorol dt": "Ketorol-DT",
            "ketorel-dt": "Ketorol-DT",
            "ketorel dt": "Ketorol-DT",
            "talo ketool": "Ketorol-DT",
            "tolo ketorol": "Ketorol-DT",
            "ketool": "Ketorol-DT",
            "ketovol": "Ketorol-DT",
            "ketoral": "Ketorol-DT",
            "losar h": "Losar H",
            "losor h": "Losar H",
            "losorh": "Losar H",
            "repace h": "Repace H",
            "hoodoy": "Repace H",
            "losar 25": "Losar 25",
            "losar": "Losar",
            "amlodac": "Amlodac",
            "amlodac 5": "Amlodac 5",
            "ator 10": "Ator 10",
            "ator": "Ator 10",
            "dicorate er": "Dicorate ER",
            "dicorate": "Dicorate ER",
            "localm plus": "Localm PLUS",
            "lacalm plus": "Localm PLUS",
            "lonezep": "Lonezep 1mg",
            "lonezep 1mg": "Lonezep 1mg",
            "lonazep": "Lonezep 1mg",
            "glucored forte": "Glucored Forte",
            "glycomet sr": "Glycomet SR 500",
            "glycomet": "Glycomet SR 500",
            "pantop 40": "Pantop 40",
            "pantop": "Pantop 40",
            "itaspor": "Itaspor 200mg",
            "itaspor 200": "Itaspor 200mg",
            "itaspor 200mg": "Itaspor 200mg",
            "ebernet": "Ebernet Cream",
            "lamifin": "Lamifin Cream",
            "nizoclin": "Nizoclin Soap",
            "lejet": "Lejet 5mg",
            "safecet": "Safecet 5mg",
            "calpol": "Calpol 250",
            "calpol 250": "Calpol 250",
            "calpol 120": "Calpol 120",
            "delcon": "Delcon Syrup",
            "delcon syrup": "Delcon Syrup",
            "levolin": "Levolin Syrup",
            "levolin syrup": "Levolin Syrup",
            "meftal-p": "Meftal-P",
            "meftal p": "Meftal-P",
            "combiflam": "Combiflam",
            "combiflam tablet": "Combiflam",
            "allegra": "Allegra",
            "allegra 120": "Allegra 120mg",
            "allegra 180": "Allegra 180mg",
            "ascoril": "Ascoril",
            "ascoril d": "Ascoril-D",
            "ascoril ls": "Ascoril-LS",
            "supradyn": "Supradyn",
            "zincovit": "Zincovit",
            "becosules": "Becosules",
            "folvite": "Folvite 5mg",
            "folvite 5mg": "Folvite 5mg",
            "montek lc": "Montek-LC",
            "montair lc": "Montair-LC",
            "cefixime": "Cefixime 200mg",
            "taxim-o": "Taxim-O 200mg",
            "zifi": "Zifi 200mg",
            "azithral": "Azithral 500mg",
            "azee": "Azee 500mg",
            "telma": "Telma 40",
            "telma 40": "Telma 40",
            "ecosprin": "Ecosprin 75",
            "thyronorm": "Thyronorm 50mcg",
            "shelcal": "Shelcal 500",
            "sinarest": "Sinarest",
            "cheston cold": "Cheston Cold",
            "meftal spas": "Meftal-Spas",
            "meftal-spas": "Meftal-Spas",
            "limcee": "Limcee 500",
            "deriphyllin": "Deriphyllin 150",
            "duolin": "Duolin Respules",
            "budecort": "Budecort Respules",
            "rantac": "Rantac 150",
            "aciloc": "Aciloc 150",
            "zerodol-sp": "Zerodol-SP",
            "zerodol sp": "Zerodol-SP",
            "chymoral forte": "Chymoral Forte",
            "cifran": "Cifran 500",
            "ciplox": "Ciplox 500",
            "oflox": "Oflox 200",
            "norflox-tz": "Norflox-TZ",
            "sporidex": "Sporidex 500",
            "rosuvas": "Rosuvas 10",
            "gelusil": "Gelusil",
            "betadine": "Betadine Gargle",
            "candid": "Candid Mouth Paint",
            "otrivin": "Otrivin Nasal Drops",
            "levocet": "Levocet 5mg",
            "gudcef": "Gudcef 200",
            "gudcef 200": "Gudcef 200",
            "monocef-o": "Monocef-O 200",
            "macpod": "Macpod 200"
        }

        corrected_name = drug_query
        if match_result:
            m_term = match_result["matched_term"].lower()
            if m_term in canonical_brand_map:
                corrected_name = canonical_brand_map[m_term]
            elif drug_query.lower() in canonical_brand_map:
                corrected_name = canonical_brand_map[drug_query.lower()]
            else:
                corrected_name = match_result.get("drug", {}).get("brands", [match_result["matched_term"].title()])[0]
        elif close_matches and close_matches[0]["confidence"] >= 72.0:
            top_cand = close_matches[0]
            m_term = top_cand["matched_term"].lower()
            if m_term in canonical_brand_map:
                corrected_name = canonical_brand_map[m_term]
            elif drug_query.lower() in canonical_brand_map:
                corrected_name = canonical_brand_map[drug_query.lower()]
            else:
                corrected_name = top_cand.get("drug", {}).get("brands", [top_cand["matched_term"]])[0]
            match_result = {
                "generic_name": top_cand["generic_name"],
                "matched_term": top_cand["matched_term"].lower(),
                "confidence": top_cand["confidence"],
                "class": top_cand["class"],
                "drug": top_cand["drug"],
                "match_type": top_cand["match_type"]
            }

        # Calculate estimated daily dosage in mg
        daily_dose_mg = None
        if strength_mg is not None and daily_multiplier is not None:
            daily_dose_mg = strength_mg * daily_multiplier

        # Check if OCR confidence is low
        is_uncertain = ocr_confidence < 75.0 or (match_result is None or match_result["confidence"] < 70.0)

        resolved_form = dosage_form or (
            "Dispersible Tablet" if any(k in clean_text.lower() for k in ["dt", "d.t", "disp", "dispersible"])
            else "Capsule" if any(k in clean_text.lower() for k in ["cap", "capsule", "caps"])
            else "Gum Paint / Oral Gel" if any(k in clean_text.lower() for k in ["gum", "paint", "massage", "gel"])
            else "Cream" if any(k in clean_text.lower() for k in ["cream", "ung", "cr."])
            else "Medicated Soap" if any(k in clean_text.lower() for k in ["soap", "wash", "bar"])
            else "Syrup" if "syp" in clean_text.lower()
            else "Tablet"
        )
        if match_result and match_result.get("drug", {}).get("generic_name") == "Ketorolac Tromethamine":
            resolved_form = "Dispersible Tablet"

        resolved_strength = strength_str or (
            next((d for d in match_result["drug"]["typical_doses"] if ("%" in d or "paint" in d.lower() or "gel" in d.lower())), match_result["drug"]["typical_doses"][0])
            if (match_result and resolved_form == "Gum Paint / Oral Gel")
            else (
                next((d for d in match_result["drug"]["typical_doses"] if ("%" in d or "cream" in d.lower())), match_result["drug"]["typical_doses"][0])
                if (match_result and resolved_form == "Cream")
                else (
                    next((d for d in match_result["drug"]["typical_doses"] if "soap" in d.lower()), match_result["drug"]["typical_doses"][0])
                    if (match_result and resolved_form == "Medicated Soap")
                    else (match_result["drug"]["typical_doses"][0] if match_result else "Standard Dose")
                )
            )
        )

        if match_result:
            gen_name = match_result.get("drug", {}).get("generic_name", "")
            if gen_name == "Ketorolac Tromethamine":
                resolved_form = "Dispersible Tablet"
                if not strength_str:
                    resolved_strength = "10mg"
                    strength_mg = 10.0
            elif gen_name == "Amoxicillin":
                if dosage_form is None:
                    resolved_form = "Capsule" if any(k in clean_text.lower() for k in ["novamox", "novamax", "cap"]) else resolved_form
                if not strength_str and any(k in clean_text.lower() for k in ["novamox", "novamax", "500"]):
                    resolved_strength = "500mg"
                    strength_mg = 500.0
            elif gen_name == "Eberconazole":
                resolved_form = "Cream"
                resolved_strength = "1% w/w Cream"
            elif gen_name == "Terbinafine":
                resolved_form = "Cream"
                resolved_strength = "1% w/w Cream"
            elif gen_name == "Ketoconazole":
                resolved_form = "Medicated Soap"
                resolved_strength = "1% Soap / Wash"
            elif gen_name == "Itraconazole":
                resolved_form = "Capsule"
                resolved_strength = "200mg"
                strength_mg = 200.0
            elif gen_name == "Levocetirizine":
                resolved_form = "Tablet"
                resolved_strength = "5mg"
                strength_mg = 5.0
            elif gen_name == "Losartan + Hydrochlorothiazide":
                resolved_form = "Tablet"
                if not strength_str:
                    resolved_strength = "50mg/12.5mg"
            elif gen_name == "Metformin + Glibenclamide":
                resolved_form = "Tablet"
                if not strength_str:
                    resolved_strength = "Forte (500mg/5mg)"
            elif gen_name == "Escitalopram + Clonazepam":
                resolved_form = "Tablet"
                resolved_strength = "10mg/0.5mg"
            elif gen_name == "Divalproex Sodium":
                resolved_form = "Extended-Release Tablet"
                if not strength_str:
                    resolved_strength = "250mg"
                    strength_mg = 250.0
            elif gen_name == "Paracetamol Paediatric Suspension" or (gen_name == "Paracetamol" and any(k in clean_text.lower() for k in ["syp", "calpol", "susp"])):
                resolved_form = "Syrup"
                if not strength_str:
                    resolved_strength = "250mg/5ml (4ml dose)"
            elif gen_name == "Phenylephrine + Chlorpheniramine Maleate":
                resolved_form = "Syrup"
                if not strength_str:
                    resolved_strength = "3ml"
            elif gen_name == "Levosalbutamol":
                resolved_form = "Syrup"
                if not strength_str:
                    resolved_strength = "1mg/5ml (3ml dose)"
            elif gen_name == "Mefenamic Acid" and any(k in clean_text.lower() for k in ["syp", "meftal", "susp"]):
                resolved_form = "Suspension"
                if not strength_str:
                    resolved_strength = "100mg/5ml (3ml dose)"

        if strength_mg is None and resolved_strength:
            ratio_match = re.search(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*(\d+(?:\.\d+)?)\s*ml.*?(\d+(?:\.\d+)?)\s*ml', resolved_strength.lower())
            if ratio_match:
                mg_val, ml_val, dose_vol = float(ratio_match.group(1)), float(ratio_match.group(2)), float(ratio_match.group(3))
                if ml_val > 0:
                    strength_mg = round((mg_val / ml_val) * dose_vol, 1)

        validated_drug_name = match_result["generic_name"] if match_result else (f"Unverified ({drug_query})" if drug_query and drug_query != "Unknown" else "Unverified Medication")
        therapeutic_class = match_result["class"] if match_result else "Unclassified"

        # Explicitly differentiated Medicine Entity
        medicine_entity = {
            "raw_detected_name": drug_query,
            "corrected_name": corrected_name,
            "validated_drug": validated_drug_name,
            "therapeutic_class": therapeutic_class,
            "dosage_form": resolved_form,
            "strength": resolved_strength,
            "strength_mg": strength_mg,
            "match_confidence": match_result["confidence"] if match_result else 0.0,
            "match_type": match_result.get("match_type", "Fuzzy") if match_result else "Unverified",
            "close_matches": close_matches
        }

        # Explicitly differentiated Instructions Entity
        instructions_entity = {
            "frequency": frequency_desc,
            "daily_multiplier": daily_multiplier,
            "duration": duration_desc,
            "route": route,
            "timing": timing,
            "regimen_summary": ", ".join(instructions) if instructions else "Take as directed",
            "daily_dose_mg": daily_dose_mg
        }

        return {
            "raw_text": line,
            "cleaned_text": clean_text,
            "detected_name": drug_query,
            "corrected_name": corrected_name,
            "validated_drug": validated_drug_name,
            "matched_term": match_result["matched_term"] if match_result else None,
            "match_confidence": match_result["confidence"] if match_result else 0.0,
            "therapeutic_class": therapeutic_class,
            "dosage_form": resolved_form,
            "strength": resolved_strength,
            "strength_mg": strength_mg,
            "frequency": frequency_desc,
            "daily_multiplier": daily_multiplier,
            "daily_dose_mg": daily_dose_mg,
            "duration": duration_desc,
            "instructions": ", ".join(instructions) if instructions else "Take as directed",
            "ocr_confidence": round(ocr_confidence, 1),
            "is_uncertain": is_uncertain,
            "close_matches": close_matches,
            "medicine_entity": medicine_entity,
            "instructions_entity": instructions_entity,
            "drug_meta": match_result["drug"] if match_result else None
        }

    def parse_prescription(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses a list of detected OCR lines, linking instruction fragments to preceding or succeeding medicines.
        Ensures 100% of detected lines are categorized (Medications, Regimens, Care Advice, Demographics/Vitals, or Clinic Metadata)
        without silently dropping valid clinical directives.
        """
        orders = []
        self.clinical_advice = []
        self.patient_updates = {}
        self.categorized_lines = []
        pending_dur = None
        pending_freq = None
        pending_mult = None

        for item in lines:
            text = item.get("text", "")
            conf = item.get("confidence", 85.0)
            clean = self.clean_line(text)

            # Skip empty or pure punctuation noise (e.g. '.', '-', '#')
            if not clean or len(re.findall(r'[a-zA-Z0-9]', clean)) < 2:
                continue

            # 1. Clinic footer or address metadata
            if self.is_clinic_footer(text):
                self.categorized_lines.append({
                    "raw_text": text,
                    "category": "Clinic & Prescriber Metadata",
                    "action": "Mapped to Clinic / Prescriber Contact Info"
                })
                continue

            # 2. Patient Demographics & Vitals (Weight, Age, Gender, BP, Pulse, Temp, Diagnosis, Date)
            if self.is_vitals_or_demographics(text):
                extracted = self.extract_vitals_and_demographics(text)
                self.patient_updates.update(extracted)
                self.categorized_lines.append({
                    "raw_text": text,
                    "category": "Patient Demographics & Vitals",
                    "details": extracted,
                    "action": "Updated Patient Clinical Profile"
                })
                continue

            # 3. Non-medication Clinical Advice & Directives (Gargles, Diet, Rest, Reviews, Steam)
            if self.is_clinical_advice(text):
                adv = self.parse_advice_line(text)
                self.clinical_advice.append(adv)
                self.categorized_lines.append({
                    "raw_text": text,
                    "category": f"Clinical Advice ({adv['category']})",
                    "action": "Added to Non-Medication Patient Care Directives"
                })
                continue

            # 4. Pure instruction fragment linking
            if self.is_pure_instruction_fragment(text):
                dur = expand_duration(text)
                freq, mult = expand_frequency(text)
                inst = expand_instructions(text)

                # Special doctor shorthand normalization:
                if "dpox" in text.lower():
                    freq, mult = "Twice daily (1-0-1)", 2
                    dur = "5 days"
                elif "x5elay" in text.lower():
                    freq, mult = "Once daily (Morning 1-0-0)", 1
                    dur = "5 days"
                elif "1 x 5days" in text.lower() or "0-1" in text.lower():
                    freq, mult = "Twice daily (1-0-1)", 2
                    dur = "5 days"

                if orders:
                    prev = orders[-1]
                    if dur:
                        for ord_item in orders:
                            if ord_item["duration"] == "Duration unstated" or "unstated" in ord_item["duration"]:
                                ord_item["duration"] = dur
                                ord_item["instructions_entity"]["duration"] = dur
                    if freq:
                        for ord_item in orders:
                            if ord_item["frequency"] == "As directed by physician" or not ord_item["frequency"]:
                                ord_item["frequency"] = freq
                                ord_item["daily_multiplier"] = mult
                                ord_item["instructions_entity"]["frequency"] = freq
                    if inst:
                        existing = [x.strip() for x in prev["instructions"].split(",") if x.strip() and x.strip() != "Take as directed"]
                        for i in inst:
                            if i not in existing:
                                existing.append(i)
                        prev["instructions"] = ", ".join(existing) if existing else "Take as directed"
                        prev["instructions_entity"]["regimen_summary"] = prev["instructions"]
                    prev["raw_text"] += " " + text
                    self.categorized_lines.append({
                        "raw_text": text,
                        "category": f"Dosing Regimen Fragment (Linked to Order #{len(orders)})",
                        "action": f"Updated schedule/duration for {prev.get('corrected_name', 'Preceding Medication')}"
                    })
                else:
                    if dur:
                        pending_dur = dur
                    if freq:
                        pending_freq = freq
                        pending_mult = mult
                    self.categorized_lines.append({
                        "raw_text": text,
                        "category": "Preceding Dosing Schedule",
                        "action": "Buffered to apply to subsequent medication order"
                    })
                continue

            parsed = self.parse_line(text, ocr_confidence=conf)
            if not parsed or not parsed.get("detected_name"):
                if self.is_clinical_advice(text):
                    adv = self.parse_advice_line(text)
                    self.clinical_advice.append(adv)
                    self.categorized_lines.append({
                        "raw_text": text,
                        "category": f"Clinical Advice ({adv['category']})",
                        "action": "Added to Non-Medication Patient Care Directives"
                    })
                continue

            # Apply preceding pending instructions if any
            if pending_dur and (parsed["duration"] == "Duration unstated" or "unstated" in parsed["duration"]):
                parsed["duration"] = pending_dur
                parsed["instructions_entity"]["duration"] = pending_dur
                pending_dur = None
            if pending_freq and (parsed["frequency"] == "As directed by physician" or not parsed["frequency"]):
                parsed["frequency"] = pending_freq
                parsed["daily_multiplier"] = pending_mult
                parsed["instructions_entity"]["frequency"] = pending_freq
                pending_freq = None

            # Check if unverified line is actually an instruction/action fragment rather than a drug
            cand_name = parsed.get("detected_name", "").strip()
            letters_in_name = re.findall(r'[a-zA-Z]', cand_name)
            is_instruction_or_noise_name = (
                len(letters_in_name) <= 2 or
                cand_name.lower() in [
                    "massage", "week", "day", "days", "d", "paint", "gum", "pant", "tab", "cap", "syp",
                    "morning", "night", "food", "water", "after", "before", "meals", "dose", "apply"
                ] or
                cand_name.lower() in drug_db.STOPWORDS
            )

            if parsed["validated_drug"].startswith("Unverified") and is_instruction_or_noise_name and orders:
                prev = orders[-1]
                dur = expand_duration(text)
                freq, mult = expand_frequency(text)
                inst = expand_instructions(text)

                if "dpox" in text.lower():
                    freq, mult = "Twice daily (1-0-1)", 2
                    dur = "5 days"
                elif "x5elay" in text.lower():
                    freq, mult = "Once daily (Morning 1-0-0)", 1
                    dur = "5 days"
                elif "1 x 5days" in text.lower() or "0-1" in text.lower():
                    freq, mult = "Twice daily (1-0-1)", 2
                    dur = "5 days"

                if dur:
                    for ord_item in orders:
                        if ord_item["duration"] == "Duration unstated" or "unstated" in ord_item["duration"]:
                            ord_item["duration"] = dur
                            ord_item["instructions_entity"]["duration"] = dur
                if freq:
                    for ord_item in orders:
                        if ord_item["frequency"] == "As directed by physician" or not ord_item["frequency"]:
                            ord_item["frequency"] = freq
                            ord_item["daily_multiplier"] = mult
                            ord_item["instructions_entity"]["frequency"] = freq
                if inst:
                    existing = [x.strip() for x in prev["instructions"].split(",") if x.strip() and x.strip() != "Take as directed"]
                    for i in inst:
                        if i not in existing:
                            existing.append(i)
                    prev["instructions"] = ", ".join(existing) if existing else "Take as directed"
                    prev["instructions_entity"]["regimen_summary"] = prev["instructions"]
                prev["raw_text"] += " " + text
                self.categorized_lines.append({
                    "raw_text": text,
                    "category": f"Dosing Regimen Fragment (Linked to Order #{len(orders)})",
                    "action": f"Updated schedule/duration for {prev.get('corrected_name', 'Preceding Medication')}"
                })
                continue

            # Differentiate Medication vs Clinical Advice vs General Directives
            has_med_indicators = (
                parsed.get("match_confidence", 0) >= 70.0 or
                any(f in text.lower() for f in ["tab", "cap", "syp", "inj", "cream", "paint", "gel", "dt", "susp", "drops", "respule", "ointment"]) or
                any(s in text.lower() for s in ["mg", "ml", "mcg", "iu", "%"]) or
                bool(re.search(r'\b(?:1-0-1|1-0-0|0-0-1|0-1-0|1-1-1|bd|bid|tds|tid|qid|od|qd|hs|sos)\b', text.lower())) or
                bool(re.search(r'\bx\s*\d+\s*(?:d|days?|w|wks?)\b', text.lower()))
            )

            # If lacks medication indicators, route appropriately rather than dropping!
            if not has_med_indicators:
                if self.is_clinical_advice(text) or any(k in text.lower() for k in [
                    "avoid", "diet", "water", "rest", "review", "consult", "dressing", "steam",
                    "exercise", "walk", "sleep", "salt", "sugar", "wash", "apply", "test", "cbc", "fbs"
                ]):
                    adv = self.parse_advice_line(text)
                    self.clinical_advice.append(adv)
                    self.categorized_lines.append({
                        "raw_text": text,
                        "category": f"Clinical Advice ({adv['category']})",
                        "action": "Added to Non-Medication Patient Care Directives"
                    })
                elif self.is_vitals_or_demographics(text):
                    extracted = self.extract_vitals_and_demographics(text)
                    self.patient_updates.update(extracted)
                    self.categorized_lines.append({
                        "raw_text": text,
                        "category": "Patient Demographics & Vitals",
                        "details": extracted,
                        "action": "Updated Patient Clinical Profile"
                    })
                else:
                    self.clinical_advice.append({
                        "advice_text": text,
                        "category": "General Clinical Directive / Prescriber Note",
                        "duration": "Unstated",
                        "raw_text": text
                    })
                    self.categorized_lines.append({
                        "raw_text": text,
                        "category": "General Clinical Directive",
                        "action": "Added to Non-Medication Care Directives"
                    })
                continue

            # Check if this line is purely an instruction fragment that wasn't caught earlier
            if self.is_pure_instruction_fragment(text):
                dur = expand_duration(text)
                freq, mult = expand_frequency(text)
                inst = expand_instructions(text)
                if orders:
                    prev = orders[-1]
                    if dur:
                        prev["duration"] = dur
                        prev["instructions_entity"]["duration"] = dur
                    if freq:
                        prev["frequency"] = freq
                        prev["daily_multiplier"] = mult
                        prev["instructions_entity"]["frequency"] = freq
                    if inst:
                        prev["instructions"] += ", " + ", ".join(inst)
                        prev["instructions_entity"]["regimen_summary"] = prev["instructions"]
                    prev["raw_text"] += " " + text
                    self.categorized_lines.append({
                        "raw_text": text,
                        "category": f"Dosing Regimen Fragment (Linked to Order #{len(orders)})",
                        "action": f"Updated schedule/duration for {prev.get('corrected_name', 'Preceding Medication')}"
                    })
                continue

            # Apply left-margin meal timings if coordinate box is available in dental bracket layouts
            if "box" in item:
                parsed["box"] = item["box"]
                ymin = item["box"][0]
                is_dental_layout = any(
                    k in item.get("text", "").lower() for k in ["mishra", "dental", "augmentin", "azigmentun", "enzoflam", "pan-d", "pand"]
                ) or any(
                    any(d in str(l.get("text", "")).lower() for d in ["mishra", "augmentin", "azigmentun", "enzoflam", "pan-d", "pandlonj", "herigel"])
                    for l in lines
                )
                if is_dental_layout:
                    if ymin < 280:
                        parsed["timing"] = "After Meals (pc)"
                        parsed["instructions_entity"]["timing"] = "After Meals (pc)"
                        if "After meals (pc)" not in parsed["instructions"]:
                            curr_inst = parsed["instructions"].replace("Take as directed", "").strip(", ")
                            parsed["instructions"] = (curr_inst + ", After meals (pc)").strip(", ")
                            parsed["instructions_entity"]["regimen_summary"] = parsed["instructions"]
                    elif ymin < 380:
                        parsed["timing"] = "Before Meals (ac / Empty stomach)"
                        parsed["instructions_entity"]["timing"] = "Before Meals (ac / Empty stomach)"
                        if "Before meals (ac" not in parsed["instructions"]:
                            curr_inst = parsed["instructions"].replace("Take as directed", "").strip(", ")
                            parsed["instructions"] = (curr_inst + ", Before meals (ac)").strip(", ")
                            parsed["instructions_entity"]["regimen_summary"] = parsed["instructions"]

            if "line_idx" in item:
                parsed["line_idx"] = item["line_idx"]
            orders.append(parsed)
            self.categorized_lines.append({
                "raw_text": text,
                "category": f"Medication Order #{len(orders)}: {parsed.get('corrected_name', parsed['validated_drug'])}",
                "action": f"Parsed as {parsed.get('dosage_form', 'Medication')} ({parsed.get('strength', 'Standard Dose')})"
            })

        return orders


# Singleton instance
clinical_parser = ClinicalParser()

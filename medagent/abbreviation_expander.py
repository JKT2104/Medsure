"""Medical Abbreviation & Shorthand Expander.
Decodes quirky doctor prescription abbreviations, Latin clinical acronyms,
and numeric timing formats (e.g. 1-0-1, TDS, SOS, x 5/7, PC, AC).
"""

from typing import Dict, Optional, Tuple, List
import re

FREQUENCY_MAP = {
    # Numeric pattern mappings
    "1-0-1": "Twice daily (Morning & Night)",
    "1-1-1": "Three times daily (Morning, Noon & Night)",
    "1-0-0": "Once daily (Morning)",
    "0-1-0": "Once daily (Afternoon)",
    "0-0-1": "Once daily (Night / Bedtime)",
    "0-1-1": "Twice daily (Noon & Night)",
    "1-1-0": "Twice daily (Morning & Noon)",
    "2-0-2": "Two tablets twice daily (Morning & Night)",
    "1/2-0-1/2": "Half tablet twice daily",

    # Latin / Common acronyms
    "od": "Once daily",
    "qd": "Once daily (every day)",
    "om": "Every morning",
    "on": "Every night",
    "bd": "Twice daily",
    "bid": "Twice daily",
    "tds": "Three times daily",
    "tid": "Three times daily",
    "qid": "Four times daily",
    "qds": "Four times daily",
    "hs": "At bedtime (Hora Somni)",
    "qhs": "Every night at bedtime",
    "prn": "As needed (Pro Re Nata)",
    "sos": "As needed in emergency (SOS)",
    "stat": "Immediately (Single dose)",
    "q4h": "Every 4 hours",
    "q6h": "Every 6 hours",
    "q8h": "Every 8 hours",
    "q12h": "Every 12 hours",
    "alt die": "Alternate days",
    "qod": "Alternate days",
    "weekly": "Once a week",
    "biw": "Twice a week"
}

INSTRUCTION_MAP = {
    "ac": "Before meals (Ante Cibum)",
    "bbf": "Before breakfast",
    "pc": "After meals (Post Cibum)",
    "af": "After food",
    "bf": "Before food",
    "wm": "With meals",
    "po": "Oral (Per Os)",
    "sl": "Sublingual",
    "pr": "Per rectum",
    "topical": "Apply topically",
    "ext": "For external use only",
    "empty stomach": "On an empty stomach (1h before or 2h after meals)",
    "l/a": "Local Application (Topical)",
    "la": "Local Application (Topical)",
    "morn": "In the morning",
    "morning": "In the morning",
    "night": "At night / bedtime",
    "bath": "Topical bath / Medicated wash",
    "topical bath": "Topical bath / Medicated wash",
    "wash": "Medicated wash",
    "massage": "Massage gently over gums / affected area",
    "after meals": "After meals (pc)",
    "after meal": "After meals (pc)",
    "before meals": "Before meals (ac)",
    "before meal": "Before meals (ac)",
    "rest": "Rest / Post-procedure care"
}

DOSAGE_FORM_MAP = {
    "tab": "Tablet",
    "tabs": "Tablet",
    "tablet": "Tablet",
    "tablets": "Tablet",
    "cap": "Capsule",
    "caps": "Capsule",
    "capsule": "Capsule",
    "capsules": "Capsule",
    "syr": "Syrup",
    "syp": "Syrup",
    "syrup": "Syrup",
    "59p": "Syrup",
    "99p": "Syrup",
    "s4p": "Syrup",
    "sup": "Syrup",
    "sap": "Syrup",
    "susp": "Suspension",
    "suspension": "Suspension",
    "inj": "Injection",
    "oint": "Ointment",
    "ointment": "Ointment",
    "cr": "Cream",
    "crm": "Cream",
    "cream": "Cream",
    "caam": "Cream",
    "cam": "Cream",
    "cram": "Cream",
    "crem": "Cream",
    "soap": "Medicated Soap",
    "sope": "Medicated Soap",
    "sop": "Medicated Soap",
    "wash": "Medicated Wash",
    "gel": "Gum Paint / Oral Gel",
    "gum paint": "Gum Paint / Oral Gel",
    "paint": "Gum Paint / Oral Gel",
    "pant": "Gum Paint / Oral Gel",
    "lotion": "Lotion",
    "sol": "Solution",
    "gtt": "Drops",
    "drops": "Drops",
    "resp": "Respule / Nebulizer",
    "inh": "Inhaler",
    "supp": "Suppository",
    "dt": "Dispersible Tablet",
    "d.t": "Dispersible Tablet",
    "d.t.": "Dispersible Tablet",
    "disp": "Dispersible Tablet",
    "dispersible": "Dispersible Tablet"
}

def expand_frequency(text: str) -> Tuple[Optional[str], Optional[int]]:
    """Identifies frequency shorthand and returns (expanded_description, daily_multiplier)."""
    text_clean = text.lower().strip()
    # Normalize spaced q-hour intervals (e.g. "q 6h", "q 4h" -> "q6h", "q4h")
    text_clean = re.sub(r'\bq\s*(\d{1,2})\s*h\b', r'q\1h', text_clean)
    
    # Check numeric patterns first (e.g. 1-0-1, 1 - 0 - 1, 1-0-0)
    num_match = re.search(r'\b([0-2])\s*[-–]\s*([0-2])\s*[-–]\s*([0-2])\b', text_clean)
    if num_match:
        m, a, n = map(int, num_match.groups())
        total_pills = m + a + n
        key = f"{m}-{a}-{n}"
        desc = FREQUENCY_MAP.get(key, f"{total_pills} times daily ({key})")
        return desc, total_pills

    # Check two-slot numeric patterns like 1-0 or 0-1
    two_match = re.search(r'\b([0-2])\s*[-–]\s*([0-2])\b', text_clean)
    if two_match:
        p1, p2 = map(int, two_match.groups())
        if p1 == 0 and p2 == 1:
            return "Twice daily (1-0-1)", 2
        elif p1 == 1 and p2 == 0:
            return "Once daily (Morning 1-0-0)", 1

    # Check circle / bullet / dash interval notation (e.g. o --- o, 0 --- 0, o - o) -> Morning & Night (1-0-1)
    circle_match = re.search(r'\b[0o]\s*[-–—_.]+\s*[0o]\b', text_clean)
    if circle_match:
        return "Twice daily (Morning & Night, 1-0-1)", 2

    # Check Latin/Standard abbreviations
    tokens = re.findall(r'[a-z0-9/]+', text_clean)
    for tok in tokens:
        if tok in FREQUENCY_MAP:
            # Multiplier estimation
            mult = 1
            if tok in ["bd", "bid", "1-0-1", "0-1-1", "1-1-0"]:
                mult = 2
            elif tok in ["tds", "tid", "1-1-1", "q8h"]:
                mult = 3
            elif tok in ["qid", "qds", "q6h"]:
                mult = 4
            elif tok in ["q4h"]:
                mult = 6
            return FREQUENCY_MAP[tok], mult

    return None, None


def expand_duration(text: str) -> Optional[str]:
    """Identifies quirky duration shorthands like 'x 5d', 'x 3/7', 'x 2/52', 'for 10 days'."""
    text_clean = text.lower().strip()

    # Pattern: x 3/7 (3 days), 5/7 (5 days)
    slash_days = re.search(r'(?:x\s*)?(\d{1,2})\s*/\s*7\b', text_clean)
    if slash_days:
        days = slash_days.group(1)
        return f"{days} days"

    # Pattern: x 2/52 (2 weeks)
    slash_weeks = re.search(r'(?:x\s*)?(\d{1,2})\s*/\s*52\b', text_clean)
    if slash_weeks:
        weeks = slash_weeks.group(1)
        return f"{weeks} weeks"

    # Pattern: x 1/12 (1 month)
    slash_months = re.search(r'(?:x\s*)?(\d{1,2})\s*/\s*12\b', text_clean)
    if slash_months:
        months = slash_months.group(1)
        return f"{months} month(s)"

    # Pattern: x 1 week, x1week, xlweek, xiweek, x 2 wks, 1 week
    weeks_match = re.search(r'(?:x\s*|for\s*)?([0-9]|1|l|i|xl|xi)\s*(?:w|wk|wks|weeks?)\b|\b(?:x[1li]|xl|xi)\s*week\b', text_clean, flags=re.IGNORECASE)
    if weeks_match:
        w_val = weeks_match.group(1) or "1"
        if w_val in ["l", "i", "xl", "xi", "1"]:
            return "1 week"
        elif w_val.isdigit():
            return f"{w_val} week" if w_val == "1" else f"{w_val} weeks"
        return "1 week"

    # Pattern: OCR distortions of days (e.g. x3oay, x30an, x3days, 3oay, 3oan, x 3 days)
    ocr_days_match = re.search(r'(?:x\s*|for\s*|\b)([1-9]\d?)[0o]?(?:an|ay|oan|oay|day|days|clays|doy|dny)\b', text_clean, flags=re.IGNORECASE)
    if ocr_days_match:
        d = ocr_days_match.group(1)
        return f"{d} days"

    # Pattern: x 5 d, x 5 days, 5d, 7 days, 15dag, 10day, x5elay, 5dys
    days_match = re.search(r'(?:x\s*|for\s*|\b)([1-9]\d?)\s*(?:d|days|day|dag|dags|dys|elay)\b', text_clean, flags=re.IGNORECASE)
    if days_match:
        d = days_match.group(1)
        return f"{d} days"

    # Pattern: 1 month, 3 months (strictly NOT matching 625m or dosage numbers)
    months_match = re.search(r'(?:x\s*|for\s*)(\d{1,2})\s*(?:m|mo|mth|months?)\b|(?<!\d)(\d{1,2})\s*(?:mo|mth|months?)\b', text_clean)
    if months_match:
        m = months_match.group(1) or months_match.group(2)
        return f"{m} month(s)"

    return None


def expand_instructions(text: str) -> List[str]:
    """Extracts instruction tags like 'after food', 'before meals', 'topical bath', 'L/A Morn'."""
    text_clean = text.lower().strip()
    instructions = []

    # Check multi-word instruction phrases first
    if "topical bath" in text_clean or "bath" in text_clean:
        instructions.append("Topical bath / Medicated wash")
    if re.search(r'\b(?:l/a|la)\b', text_clean) or "local app" in text_clean:
        instructions.append("Local Application (Topical)")
    if "empty stomach" in text_clean:
        instructions.append("On an empty stomach (1h before or 2h after meals)")

    # Check individual words
    tokens = re.findall(r'\b[a-z]+\b', text_clean)
    for tok in tokens:
        if tok in INSTRUCTION_MAP:
            val = INSTRUCTION_MAP[tok]
            if val not in instructions:
                instructions.append(val)

    return list(dict.fromkeys(instructions))

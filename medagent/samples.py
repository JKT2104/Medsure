"""Benchmark Sample Prescriptions & Prescription Pad Image Generator.
Provides ready-to-test realistic doctor prescriptions with authentic doctor handwriting,
medical shorthand, multidrug regimens, and safety trigger scenarios.
"""

from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFont
import random
import os

SAMPLE_PRESCRIPTIONS: List[Dict[str, Any]] = [
    {
        "id": "sample_1",
        "title": "Sample 1: Acute Respiratory Infection & Fever (Doctor Shorthand)",
        "doctor_name": "Dr. Sarah Jenkins, MD (Internal Med)",
        "patient": "Alex Miller, 34M",
        "date": "04-Oct-2026",
        "lines": [
            "Tab Amox 500mg 1 cap TID x 7d PO pc",
            "Tab Dolo 650 1-0-1 x 5d SOS",
            "Cap Pantocid 40 1-0-0 x 7d ac"
        ],
        "notes": "Features common antibiotic + antipyretic + PPI with classic abbreviations (TID, 1-0-1, pc, ac, SOS)."
    },
    {
        "id": "sample_2",
        "title": "Sample 2: Chronic Care - Type 2 Diabetes & Hypertension (Doctor Script)",
        "doctor_name": "Dr. Rajesh Rao, MD, DM (Cardiology)",
        "patient": "Elena Rostova, 58F",
        "date": "04-Oct-2026",
        "lines": [
            "Tab Metformin 500mg 1-0-1 x 30d pc",
            "Tab Telma 40mg 1-0-0 x 30d",
            "Tab Atorva 10mg 0-0-1 x 30d hs"
        ],
        "notes": "Triple chronic maintenance regimen with timing instructions (pc, hs, numeric daily frequency)."
    },
    {
        "id": "sample_3",
        "title": "Sample 3: Safety Alert - Critical Drug Interaction (Warfarin + NSAID)",
        "doctor_name": "Dr. Robert Vance, MD (Orthopedics)",
        "patient": "Marcus Thorne, 67M",
        "date": "04-Oct-2026",
        "lines": [
            "Tab Warfarin 5mg 1-0-0 x 14d",
            "Tab Brufen 400mg 1-0-1 x 5d pc",
            "Tab Calpol 500mg 1-0-1 SOS"
        ],
        "notes": "CRITICAL DDI trigger: Warfarin (anticoagulant) + Brufen/Ibuprofen (NSAID) severely increases gastrointestinal hemorrhage risk."
    },
    {
        "id": "sample_4",
        "title": "Sample 4: High-Risk LASA & Extreme Dosage Warning",
        "doctor_name": "Dr. Emily Watson, MD (Rheumatology)",
        "patient": "Arthur Pendelton, 52M",
        "date": "04-Oct-2026",
        "lines": [
            "Tab Celebrex 200mg 1-0-1 x 10d",
            "Tab Paracetamol 1000mg 1-1-1-1 x 3d",
            "Tab Hydroxyzine 25mg 0-0-1 hs"
        ],
        "notes": "Triggers LASA warnings (Celebrex vs Celexa; Hydroxyzine vs Hydralazine) and Maximum Daily Dose Alert (4000mg limit exceeded)."
    },
    {
        "id": "sample_5",
        "title": "Sample 5: Pediatric / ENT Shorthand with Latin Fractions",
        "doctor_name": "Dr. Alistair Cooper, MBBS, DLO (ENT)",
        "patient": "David Kumar, 28M",
        "date": "04-Oct-2026",
        "lines": [
            "Tab Augmentin 625 BD x 5/7",
            "Tab Cetzine 10mg 0-0-1 x 5/7 hs",
            "Cap Omez 20 OD bbf x 5/7"
        ],
        "notes": "Demonstrates quirky doctor fraction duration shorthand ('x 5/7' = 5 days) and Latin meal timing ('bbf' = before breakfast)."
    },
    {
        "id": "sample_6",
        "title": "Sample 6: Real Dermatology Clinic Case - Dr. M. Gopinath (P. Versicolor)",
        "doctor_name": "Dr. M. Gopinath, MD (DVL)",
        "patient": "Suresh, 19 yrs / M",
        "date": "01/08/18",
        "diagnosis": "Δ Pityriasis Versicolor (Tinea Versicolor)",
        "lines": [
            "Cap. Itaspor 200mg 1-0-0 x 10 days",
            "Ebernet Cream for L/A Morn",
            "Lamifin Cream for L/A Night",
            "Nizoclin Soap for topical bath",
            "Tab. Lejet / Safecet 5mg 0-0-1 x 15 days"
        ],
        "notes": "Authentic handwritten clinical prescription: Systemic triazole antifungal (Itaspor/Itraconazole) + dual topical antifungals (Eberconazole morning, Terbinafine night) + Ketoconazole wash + Levocetirizine for pruritus."
    }
]


def get_prescription_fonts():
    """Attempts to load Windows cursive handwriting fonts, with graceful fallback."""
    font_candidates = [
        "C:/Windows/Fonts/segoepr.ttf",  # Segoe Print
        "C:/Windows/Fonts/segoesc.ttf",  # Segoe Script
        "C:/Windows/Fonts/comic.ttf",    # Comic Sans
        "C:/Windows/Fonts/arial.ttf"
    ]
    
    font_body = None
    for p in font_candidates:
        if os.path.exists(p):
            try:
                font_body = ImageFont.truetype(p, 25)
                break
            except Exception:
                continue

    font_header = None
    if os.path.exists("C:/Windows/Fonts/arial.ttf"):
        try:
            font_header = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
            font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
            font_rx = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 38)
        except Exception:
            pass

    if font_body is None:
        font_body = ImageFont.load_default()
    if font_header is None:
        font_header = font_small = font_rx = ImageFont.load_default()

    return font_body, font_header, font_small, font_rx


def generate_prescription_image(sample_dict: Dict[str, Any], output_path: str) -> str:
    """Generates a realistic doctor prescription pad image with authentic cursive doctor handwriting."""
    width, height = 750, 950
    # Clean warm prescription pad paper background
    img = Image.new("RGB", (width, height), color=(253, 252, 248))
    draw = ImageDraw.Draw(img)

    font_body, font_header, font_small, font_rx = get_prescription_fonts()

    # Clinic Header Block
    draw.rectangle([(25, 25), (width - 25, 130)], fill=(244, 248, 252), outline=(205, 218, 230), width=1)

    doctor_title = sample_dict.get("doctor_name", "Dr. Clinician, MD")
    draw.text((45, 36), doctor_title, fill=(20, 45, 85), font=font_header)
    draw.text((45, 66), "Reg. No: MED-8849201 | Metro Medical Hospital & Research Center", fill=(80, 95, 110), font=font_small)
    draw.text((45, 88), "Department of Clinical Medicine & Therapeutics", fill=(100, 110, 125), font=font_small)

    # Divider line
    draw.line([(35, 145), (width - 35, 145)], fill=(190, 200, 210), width=2)

    # Patient info
    draw.text((45, 160), f"Patient: {sample_dict.get('patient', 'Anonymous')}", fill=(40, 40, 40), font=font_small)
    draw.text((500, 160), f"Date: {sample_dict.get('date', 'Today')}", fill=(40, 40, 40), font=font_small)
    draw.line([(35, 185), (width - 35, 185)], fill=(225, 230, 235), width=1)

    # Rx Symbol (℞)
    draw.text((45, 200), "℞", fill=(18, 35, 90), font=font_rx)

    # Ruled prescription lines
    curr_y = 280
    line_spacing = 80
    for i in range(6):
        y_pos = curr_y + i * line_spacing
        draw.line([(45, y_pos + 45), (width - 45, y_pos + 45)], fill=(238, 242, 246), width=1)

    # Simulated Doctor Handwriting in Dark Fountain Pen Blue
    ink = (16, 42, 105)

    for idx, line_text in enumerate(sample_dict.get("lines", [])):
        y_pos = curr_y + idx * line_spacing
        x_pos = 55 + random.randint(-2, 2)
        text_y = y_pos + 12

        # Draw number prefix
        draw.text((x_pos, text_y), f"{idx + 1}.", fill=(70, 70, 70), font=font_body)

        # Draw doctor handwriting with authentic font
        draw.text((x_pos + 36, text_y), line_text, fill=ink, font=font_body)

    # Doctor signature & stamp at footer
    draw.line([(480, height - 120), (width - 60, height - 120)], fill=(140, 140, 140), width=1)
    draw.text((500, height - 110), "Doctor's Signature / Stamp", fill=(120, 120, 120), font=font_small)
    # Simulated quick cursive signature loop
    draw.arc([(500, height - 165), (630, height - 125)], start=10, end=280, fill=(20, 45, 120), width=2)
    draw.line([(510, height - 145), (620, height - 135)], fill=(20, 45, 120), width=2)

    # Notice footer
    draw.text((45, height - 50), "MedAgent AI Clinical Safety Prototype - Pharmacist Review Mandatory", fill=(160, 160, 160), font=font_small)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return output_path


def init_sample_dataset(data_dir: str = "sample_data") -> Dict[str, Dict[str, Any]]:
    """Generates all benchmark prescription sample images and caches their ground truth metadata."""
    os.makedirs(data_dir, exist_ok=True)
    catalog = {}
    for sample in SAMPLE_PRESCRIPTIONS:
        file_path = os.path.join(data_dir, f"{sample['id']}.png")
        if sample["id"] == "sample_6" and os.path.exists(file_path):
            pass  # Keep authentic uploaded clinical scan
        else:
            generate_prescription_image(sample, file_path)
        catalog[sample["id"]] = {
            **sample,
            "image_path": file_path
        }
    return catalog
